---
layout:     post
title:      Tomcat调用CDH Spark分布式计算的框架实现
subtitle:   相关总结
date:       2018-06-07
author:     HC
header-img: img/post-bg-miui6.jpg
catalog: true
tags:
    - Spark on Yarn
    - CDH
    - Tomcat
    - Redis
---

> 项目的坑，还是要写一下
>
> 未完待续

# 0. 前言

我们的整个分布式计算平台建立在Cloudera CDH 5.10.1之上，所用到的组件包括（其中tomcat和redis不在CDH中）：

1. hadoop 2.6.0-cdh5.10.1
2. elasticsearch 5.4.0
3. scala 2.10.5
4. spark 1.6.0-cdh5.10.1
5. spark-redis 0.3.2
6. tomcat 8

项目功能需要实现通过web调用spark进行任务计算，并返回计算结果。结合项目的具体要求，web端我们采用了tomcat，再提交Spark计算请求，计算结果缓存在Redis内存数据库中，待controller取用。

整个过程，躺枪太多，下面是相关总结。

# 1. Spark的调用方法总结

这部分是躺枪最严重的地方，遇到了很多莫名其妙的bug。总的来说，我们试过了下面四种Spark的任务提交方式。前三种方法坑太多，至今还有bug。最后我们采用了第4中提交方式。

1. Shell 提交
2. SparkSubmit
3. Client submit
4. SparkLauncher


## 1.1 Shell Submit

在命令行里进行提交计算任务是最简单的方式，直接调用spark-submit命令就可以了，如下所示

```shell
spark-submit \
  --class <main-class> \
  --master <master-url> \
  --deploy-mode <deploy-mode> \
  --conf <key>=<value> \
  ... # other options
  <application-jar> \
  [application-arguments]
```

由于我们将tomcat安装在了集群主节点上，所以，一个最偷懒的方式就是在Spring Controller中调用java runtime来执行spark-submit命令。所以，我们使用了``getCMDParams``来构造命令，然后使用``Runtime.getRuntime().exec(cmd)``来执行Spark任务。注意，这提交方式无法获取application id，也就无法获取到任务的状态（比如ACCEPT, RUNNING, FINISHED, KILLED, FAILED等等）

```java
public void submitByCMD(ArrayList<String> args, boolean enableLog){
        String[] cmd = getCMDParams(args);
        log.info("run cmd: " + String.join(" ", cmd));
        try{
            Process child = Runtime.getRuntime().exec(cmd);
            //child.waitFor();
            if(enableLog){
                log.info("input stream:");
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(child.getInputStream()));
                String line = bufferedReader.readLine();
                while(line != null){
                    log.debug(line);
                    line = bufferedReader.readLine();
                }
                log.info("error stream:");
                bufferedReader = new BufferedReader(new InputStreamReader(child.getErrorStream()));
                line = bufferedReader.readLine();
                while(line != null){
                    log.debug(line);
                    line = bufferedReader.readLine();
                }
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }
///////////////////////////////我是分割线///////////////////////////////////////////////
public String[] getCMDParams(ArrayList<String> algParams){
        ArrayList<String> lists = new ArrayList<>();
        if(master == null || master.length() == 0 || jar.length() < 1 || className.length() < 1){
            return null;
        }
        lists.add("spark-submit");
        lists.add("--master");
        lists.add(master);

        lists.add("--name");
        lists.add(jobName);

        lists.add("--class");
        lists.add(className);
        if(executorMemory.length() > 0){
            lists.add("--executor-memory");
            lists.add(executorMemory);
        }
        if(numExecutors.length() > 0){
            lists.add("--num-executors");
            lists.add(numExecutors);
        }

        if(executorCores.length() > 0){
            lists.add("--executor-cores");
            lists.add(executorCores);
        }

        lists.add("--driver-memory");
        lists.add("32G");

        lists.add("--jars");
        lists.add(addJars);
        lists.add(jar);
        if(algParams != null || algParams.size() > 0){
            lists.addAll(algParams);
        }
        return (String[]) lists.toArray(new String[0]);
    }
```

经过实验，这种方式是可以提交成功的。一个可能存在的潜在bug就是Runtime的内存使用了，我们不希望在执行Spark任务的时候撑爆Runtime或者Tomcat的运行内存。所以，我们将Tomcat的执行内存进行了扩大：在Tomcat的bin文件夹下，新建``setenv.sh``文件，填入下面的东东，重启Tomcat生效

```shell
export CATALINA_OPTS="$CATALINA_OPTS -Xms4096m"
export CATALINA_OPTS="$CATALINA_OPTS -Xmx12288m"
export CATALINA_OPTS="$CATALINA_OPTS -XX:MaxPermSize=512m"
```

但是，人这辈子不可能一帆风顺。这种方法提交的计算任务会卡死在action操作上，报错：``org.apache.spark.rpc.RpcTimeoutException: Futures timed out after [10 seconds]. This timeout is controlled by spark.rpc.lookupTimeout`` 

我们首先试着把timeout的时间阈值加大。即在提交的时候增加参数，把时间扩大到5分钟。但是还是会timeout

```shell
lists.add("--conf");
lists.add("spark.executor.heartbeatInterval=300s");
lists.add("--conf");
lists.add("spark.network.timeout=300s");
```

同时，我们也通过增加driver memory，减少所处理的数据量的方式排除内存溢出的问题，但是这个bug依然存在。无解哎。

## 1.2 SparkSubmit

Spark里提供了一个SparkSubmit的类，以供大家调用。它与Runtime的调用方法比较类似，直接用``SparkSubmit.main(cmd)``就可以了。之所以说比较类似，是因为这个方式提交的application同样会卡死在action上，一直timeout。并且也不能获取application的状态

```java
public void submitBySpark(ArrayList<String> args){
        String[] cmd = getSparkParams(args);
        log.info("run cmd: " + String.join(" ", cmd));
        SparkSubmit.main(cmd);
    }
```

而且，SparkSubmit调用方法更坑的是，他需要手动给定运行的环境变量（难道CDH是吃素的吗？咋SparkSubmit自动获取不到环境变量呢）。我们在``getSparkParams(args)``中指定了大量的环境参数：

```java
public String[] getSparkParams(ArrayList<String> algParams){
        ArrayList<String> lists = new ArrayList<>();
        if(master == null || master.length() == 0 || jar.length() < 1 || className.length() < 1){
            return null;
        }
        lists.add("--master");
        lists.add(master);

        lists.add("--name");
        lists.add(jobName);

        lists.add("--class");
        lists.add(className);
        if(executorMemory.length() > 0){
            lists.add("--executor-memory");
            lists.add(executorMemory);
        }
        if(numExecutors.length() > 0){
            lists.add("--num-executors");
            lists.add(numExecutors);
        }

        if(executorCores.length() > 0){
            lists.add("--executor-cores");
            lists.add(executorCores);
        }

        lists.add("--conf");
        lists.add("spark.executor.heartbeatInterval=60s");
        lists.add("--conf");
        lists.add("spark.network.timeout=60s");
        lists.add("--conf");
        lists.add("spark.yarn.resourcemanager.address=hadoop11:8032");
        lists.add("--conf");
        lists.add("spark.yarn.resourcemanager.scheduler.address=hadoop11:8030");
        lists.add("--conf");
        lists.add("spark.yarn.resourcemanager.resource-tracker.address=hadoop11:8031");
        lists.add("--conf");
        lists.add("spark.yarn.mapreduce.jobhistory.address=hadoop11:10020");
        lists.add("--conf");
        lists.add("spark.fs.defaultFS=hdfs://hadoop11:8020");
        lists.add("--conf");
        lists.add("spark.yarn.application.classpath=/etc/hadoop/conf:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/./:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/.//*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/lib/*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/.//*:");
        lists.add("--conf");
        lists.add("spark.home=/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/spark");
        lists.add("--conf");
        lists.add("spark.yarn.jar=hdfs://hadoop11:8020/lib/spark-assembly-1.6.0-cdh5.10.1-hadoop2.6.0-cdh5.10.1.jar");
        lists.add("--conf");
        lists.add("spark.yarn.config.replacementPath=/etc/spark/conf.cloudera.spark_on_yarn");
        lists.add("--conf");
        lists.add("spark.yarn.config.gatewayPath=/etc/spark/conf.cloudera.spark_on_yarn");
        lists.add("--conf");
        lists.add("spark.yarn.appMasterEnv.HADOOP_HOME=/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop");
        lists.add("--conf");
        lists.add("spark.yarn.appMasterEnv.HADOOP_CONF_DIR=/etc/hadoop/conf");
        lists.add("--conf");
        lists.add("spark.yarn.appMasterEnv.YARN_CONF_DIR=/etc/hadoop.cloudera.yarn");
        lists.add("--conf");
        lists.add("spark.yarn.appMasterEnv.SPARK_CONF_DIR=/etc/spark/conf.cloudera.spark_on_yarn");
        lists.add("--conf");
        lists.add("spark.yarn.appMasterEnv.LD_LIBRARY_PATH=/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/lib/native");
        lists.add("--conf");
        lists.add("spark.yarn.appMasterEnv.SPARK_DIST_CLASSPATH=/etc/hadoop/conf:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/./:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/.//*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/lib/*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/.//*:");
        lists.add("--conf");
        lists.add("spark.driver.extraLibraryPath=/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/lib/native");
        lists.add("--conf");
        lists.add("spark.hadoop.yarn.application.classpath=/etc/hadoop/conf:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/./:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/.//*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/lib/*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/.//*:");
        lists.add("--jars");
        lists.add(addJars);
        lists.add("--properties-file");
        lists.add("/etc/spark/conf/spark-defaults.conf");
        lists.add(jar);
        if(algParams != null || algParams.size() > 0){
            lists.addAll(algParams);
        }
        return (String[]) lists.toArray(new String[0]);
    }
```

手动设置环境变量比较坑，每次都是遇到一个bug之后才知道要添加哪个环境变量。比如：

1. 遇到hadoop compress方面的bug，比如下面，就考虑是否指定了``hadoop/lib/native``。

   > java.lang.UnsatisfiedLinkError: org.apache.hadoop.util.NativeCodeLoader.buildSupportsSnappy()Z

2. 还有一个比较坑的是下面这种报错

   > Diagnostics: File file:/usr/local/apache-tomcat-7.0.82/temp/spark-bb37673d-d2e2-415e-815f-2f55f6466e9b/__spark_conf__68442712017931812.zip does not exist
   >
   > java.io.FileNotFoundException: File file:/usr/local/apache-tomcat-7.0.82/temp/spark-bb37673d-d2e2-415e-815f-2f55f6466e9b/__spark_conf__68442712017931812.zip does not exist

   从字面上看是文件不存在，但是主节点上有这个文件，但是这个配置文件本应该被上传到HDFS中，以供其他节点访问，但是报错中确是一个local file。出现这个问题的原因是Spark配置缺少hadoop的相关信息（哎，找不到hadoop，还上传什么鬼哦）。所以加上呗：

   ```java
   Configuration conf = new Configuration()
   conf.set("yarn.resourcemanager.address","hadoop11:8032");
   conf.set("yarn.resourcemanager.scheduler.address", "hadoop11:8030");
   conf.set("yarn.resourcemanager.resource-tracker.address", "hadoop11:8031");
   conf.set("fs.defaultFS", "hdfs://hadoop11:8020");
   ```

   在SparkConf中设置Hadoop相关的参数的话，要在前面加``spark``，比如hadoop conf是yarn.resourcemanager.address的话，在sparkconf中就是spark.yarn.resourcemanager.address

3. 另外，也可能会报错说找不到hadoop的 Configuration这个类，这时候需要格外指明hadoop classpath，也就是``yarn.application.classpath``。

4. 其他报错，不记得了。。。总之很多

## 1.3 Client Submit

Spark Client的提交方式可以通过application id来追踪application的运行状态。这个Client是对hadoop 中YarnClient这个类的一个封装。提交方式如下：

```java
public String submitByClientWithMonitor(ArrayList<String> algParams){
        String[] arg = getParams(algParams, false);
        log.info("run cmd:" + String.join(" ", arg));
        if(arg == null)
            return "-1";
        Client client = getClient(arg);
        ApplicationId appId = null;
        try{
            appId = client.submitApplication();
        }catch(Throwable e){
            e.printStackTrace();
            return "-1";
        }
        Utils.updateAppStatus(appId.toString(), "2%" );
        log.info(Utils.allAppStatus.toString());
        new Thread(new CheckProcess(appId, client)).start();
        return appId.toString();
    }
```

在获取到applicationid之后，可以通过下面的方式获取到程序的当前运行状态

```java
YarnApplicationState state = client.getApplicationReport(appId).getYarnApplicationState();
```

正常情况下application的状态有以下几种取值。当达到applicationid的时候，已经表明application被提交成功了，直接就进入了accepted的状态了。有了这个我们就可以对application的进度进行监控。

1. YarnApplicationState.ACCEPTED
2. YarnApplicationState.RUNNING
3. YarnApplicationState.FINISHED
4. YarnApplicationState.FAILED
5. YarnApplicationState.KILLED

一个比较麻烦的是Client的生成，又需要手动的添加很多环节变量。这里他需要使用SparkConf和Hadoop Configuration一起来生成Client，所以对于hadoop的相关配置可以单独在configuration中进行设置。

```java
public Client getClient(String[] arg){
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("yarn");
        sparkConf.set("mapreduce.framework.name","yarn");
        sparkConf.set("mapreduce.jobhistory.address","hadoop11:10020");
        //sparkConf.set("spark.yarn.preserve.staging.files", "true");
        sparkConf.set("spark.home", "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/spark");
        sparkConf.set("spark.yarn.jar", "hdfs://hadoop11:8020/lib/spark-assembly-1.6.0-cdh5.10.1-hadoop2.6.0-cdh5.10.1.jar");
        sparkConf.set("spark.yarn.config.replacementPath", "/etc/spark/conf.cloudera.spark_on_yarn");
        sparkConf.set("spark.yarn.config.gatewayPath", "/etc/spark/conf.cloudera.spark_on_yarn");
        sparkConf.set("spark.yarn.appMasterEnv.HADOOP_HOME", "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop");
        sparkConf.set("spark.yarn.appMasterEnv.HADOOP_CONF_DIR", "/etc/hadoop/conf");
        sparkConf.set("spark.yarn.appMasterEnv.YARN_CONF_DIR", "/etc/hadoop.cloudera.yarn");
        sparkConf.set("spark.yarn.appMasterEnv.SPARK_CONF_DIR", "/etc/spark/conf.cloudera.spark_on_yarn");
        sparkConf.set("spark.yarn.appMasterEnv.LD_LIBRARY_PATH", "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/lib/native");
        sparkConf.set("spark.yarn.appMasterEnv.SPARK_DIST_CLASSPATH", "/etc/hadoop/conf:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/./:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/.//*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/lib/*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/.//*:" +
                "/usr/local/elasticsearch-5.4.0/lib/elasticsearch-spark-13_2.10-5.4.0.jar:" +
                "/usr/local/elasticsearch-5.4.0/lib/lingdian_mini.jar");
        sparkConf.set("spark.driver.extraLibraryPath", "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/lib/native");
        sparkConf.set("spark.hadoop.yarn.application.classpath", "/etc/hadoop/conf:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/./:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/.//*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/lib/*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/.//*:" +
                "/usr/local/elasticsearch-5.4.0/lib/elasticsearch-spark-13_2.10-5.4.0.jar:" +
                "/usr/local/elasticsearch-5.4.0/lib/lingdian_mini.jar");

        System.setProperty("SPARK_YARN_MODE", "true");
        System.setProperty("HADOOP_CONF_DIR", "/etc/hadoop/conf");
        System.setProperty("YARN_CONF_DIR", "/etc/hadoop.cloudera.yarn");
        System.setProperty("SPARK_CONF_DIR", "/etc/spark/conf.cloudera.spark_on_yarn");

        ClientArguments cArgs = new ClientArguments(arg, sparkConf);
        Configuration conf = new Configuration();
        conf.set("yarn.resourcemanager.address","hadoop11:8032");
        conf.set("yarn.resourcemanager.scheduler.address", "hadoop11:8030");
        conf.set("yarn.resourcemanager.resource-tracker.address", "hadoop11:8031");
        conf.set("fs.defaultFS", "hdfs://hadoop11:8020");
        conf.set("yarn.application.classpath", "/etc/hadoop/conf:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/./:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-hdfs/.//*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/lib/*:" +
                "/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/hadoop/libexec/../../hadoop-yarn/.//*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/lib/*:" +
                "/opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/.//*:" +
                "/usr/local/elasticsearch-5.4.0/lib/elasticsearch-spark-13_2.10-5.4.0.jar:" +
                "/usr/local/elasticsearch-5.4.0/lib/lingdian_mini.jar");
        return new Client(cArgs,conf,sparkConf);
    }
```

另外一个需要注意的地方是，Client提交时所用的命令参数是有点不同于Spark-submit的方式。

```java
public String[] getParams(ArrayList<String> algParams, boolean addMaster){
        ArrayList<String> lists = new ArrayList<>();
        if(master == null || master.length() == 0 || jar.length() < 1 || className.length() < 1){
            return null;
        }
        if(addMaster){
            lists.add("--master");
            lists.add(master);
        }

        lists.add("--name");
        lists.add(jobName);

        lists.add("--class");
        lists.add(className);
        if(executorMemory.length() > 0){
            lists.add("--executor-memory");
            lists.add(executorMemory);
        }
        if(numExecutors.length() > 0){
            lists.add("--num-executors");
            lists.add(numExecutors);
        }

        if(executorCores.length() > 0){
            lists.add("--executor-cores");
            lists.add(executorCores);
        }

        lists.add("--jar"); //程序的jar
        lists.add(jar);
  
		//需要配合使用的jar，必须是local的jar，不能是在hdfs上，否则报错。
  		//并且同一个jar不能同时设置--addJars和--archives，否则还是会报错
        lists.add("--addJars"); 
        lists.add(addJars);

        if(algParams != null || algParams.size() > 0){
            for(String s:algParams){
                lists.add("--arg"); //程序运行参数
                lists.add(s);
            }
        }

        return (String[]) lists.toArray(new String[0]);
    }
```

在使用过程中，我们遇到的唯一bug是，尽管application运行正常，但是application的状态一直处理accepted的状态。整个过程一直都没有变成running的状态，直到程序运行结束，application的状态才变为finished（或者failed如果有错的话）。

## 1.4 SparkLauncher

最后的最后，我们采用了SparkLauncher来提交作业，他实现简单，耶可以追踪application的运行状态。在我们的环境里，唯一需要在程序里指定的是SparkHome

```java
public void submitByLang(ArrayList<String> args){
        SparkLauncher launcher = new SparkLauncher();
        launcher.setAppName(jobName);
        launcher.setMaster("yarn");
        launcher.setAppResource(jar); //主程序的jar
        launcher.setMainClass(className);
        launcher.addJar("/usr/local/elasticsearch-5.4.0/lib/elasticsearch-spark-13_2.10-5.4.0.jar"); //需要搭配使用的jar
        launcher.setSparkHome("/opt/cloudera/parcels/CDH-5.10.1-1.cdh5.10.1.p0.10/lib/spark");
        launcher.setConf(SparkLauncher.DRIVER_MEMORY, "32g");
        launcher.setConf(SparkLauncher.EXECUTOR_MEMORY, "60g");
        launcher.setConf(SparkLauncher.EXECUTOR_CORES, "30");
        //jar包路径为在整个文件系统中的路径。
        launcher.addAppArgs(args.toArray(new String[0]));//运行参数
        launcher.setVerbose(true);
        SparkAppHandle handle = null;
        try {
            handle = launcher.startApplication();  //简单的获取application的进度
            while(!(handle.getState() == SparkAppHandle.State.FINISHED ||
                    handle.getState() == SparkAppHandle.State.FAILED ||
                    handle.getState() == SparkAppHandle.State.KILLED)) {
                Thread.sleep(3000L);
                System.out.println("applicationId is: "+ handle.getAppId());
                System.out.println("current state: "+ handle.getState());
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }
```

这是唯一一个在我们环境下能正常运行的方法。

# 2. Tomcat & ElasticSearch & Spark的包冲突

第二个大坑应该就是包冲突的问题了。主要有两个冲突，

1. spark-assembly-1.6.0-cdh5.10.1-hadoop2.6.0-cdh5.10.1.jar和Tomcat中的servlet-api之间的冲突 
2. Tomcat和elasticsearch-spark-13_2.10-5.4.0.jar之间的冲突

## 2.1 spark-assembly 

使用SparkSubmit和Client方式的时候，需要指定``spark.yarn.jar``为spark-assembly的jar包（这里可以把这个包放在hdfs上，避免使用local的包，在运行的时候spark会费时上传到hdfs上）。但是spark-assembly里有servlet，会和tomcat冲突。这里我们参照了网上的方法：

1. 弄两份spark-assembly，一份是原始的，记为A，把另一份中的servlet文件夹删除，记为B
2. 把A放在tomcat的WEB的lib中，把B放在hdfs上

## 2.2 elasticsearch-spark 

在使用ElasticSearch的时候，我们遇到报错：``ClassNotFoundException:org.elasticsearch.spark.rdd.EsPartition``。我们按照网上的方案把ElasticSearch的相关包放在了Spark的jars下，但是问题并没有解决。猜测是和tomcat有冲突。这里我们采用的是在提交任务的时候，用addjar的命令加入了这个包，从而绕过这个报错。