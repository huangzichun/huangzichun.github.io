---
layout:     post
title:      在Ubuntu系统下伪分布式安装Hadoop，Spark和Hive
subtitle:   Hadoop, Spark, Mysql, Hive
date:       2018-03-26
author:     HC
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - Hadoop
    - Spark
    - Hive
    - Mysql
    - Ubuntu
---

> 入坑前的准备工作

# 0. 安装所需文件

1. Hadoop-3.0.0.tar.gz
2. Scala-2.12.5.tar.gz
3. Spark-2.3.0.tar.gz
4. Hive-2.3.2.tar.gz
5. Java 1.8.0_162.tar.gz
6. mysql-connector-java-5.1.46.jar

# 1. 安装Hadoop

## 1.1 安装jdk

解压下载好的jdk，放置在你想放的目录下。然后编辑/etc/profile，配置环境变量。

```shell
tar -zxvf $YOUR_JDK_FILE -C $OUTPUT_PATH
sudo vim /etc/profile
	# 添加下面的环境变量
	export JAVA_HOME=/usr/local/jdk1.8.0_162
	export JRE_HOME=$JAVA_HOME/jre
	export CLASS_PATH=.:$JAVA_HOME/lib/tools.jar:$JAVA_HOME/lib/dt.jar
	export PATH=$PATH:$JAVA_HOME/bin
source /etc/profile
```

执行完了之后，可以输入java -version试一试。注意的是，强烈推荐不要用java 10，好像java 10和hadoop 3.0.0有兼容问题。

## 1.2 设置SSH免密码登录

安装SSH

```shell
sudo apt-get install openssh-server
```

用ssh生成自己的密钥对。这里我采用了dsa的加密方式

```shell
ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa
cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
```

如果需要配置分布式环境，则需要每个节点都生成一份密钥对。然后将NameNode的公钥复制到各个节点上。并且把他追加到authorized_keys中。

## 1.3 Hadoop安装与配置

解压hadoop的压缩包到你喜欢的地方。添加环境变量

```shell
tar -zxvf $YOUR_HADOOP_FILE -C $HADOOP_OUTPUT_PATH
sudo vim /etc/profile
	# 添加
	export HADOOP_HOME=/usr/local/hadoop-3.0.0
	export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin
source /etc/profile
```

接下来，需要对Hadoop对相关配置。配置文件在hadoop文件夹下的etc/hadoop子文件夹中，需要配置的文件有hadoop-env.sh, hdfs-site.xml, core-site.xml, mapred-site.xml和yarn-site.xml。我们需要在hadoop-env.sh文件中加入JAVA_HOME的信息：

```shell
export JAVA_HOME=$YOUR_JAVA_HOME
```

在hdfs-site.xml文件中添加一下信息。我的HADOOP_HOME是/usr/local/hadoop-3.0.0/。确保HADOOP_HOME下存在hdfs/name和hdfs/data两个文件夹，否则自行新建

```xml
<configuration>
<property>
	<name>dfs.replication</name>
	<value>1</value>
</property>
<property>
    <name>dfs.permissions</name>
    <value>false</value>
</property>
<property>  
    <name>dfs.namenode.name.dir</name>  
    <value>file:///usr/local/hadoop-3.0.0/hdfs/name</value>  
</property> 
<property>  
     <name>dfs.datanode.data.dir</name>  
     <value>file:///usr/local/hadoop-3.0.0/hdfs/data</value>  
</property>  
<property>  
	<name>dfs.namenode.secondary.http-address</name>  
	<value>localhost:9001</value>  
</property>  
</configuration>
```

在core-site.xml中加入下面的信息。同样保证HADOOP_HOME下，存在tmp文件夹

```xml
<configuration>
<property>  
	<name>fs.defaultFS</name>   
	<value>hdfs://localhost:9000</value>  
</property>       
<property>  
	<name>hadoop.tmp.dir</name>
	<value>file:///usr/local/hadoop-3.0.0/tmp</value>  
</property> 
</configuration>
```

mapred-site.xml文件中加入

```xml
<configuration>
<property>
	<name>mapreduce.framework.name</name>
	<value>yarn</value>
</property>
<property>
	<name>yarn.app.mapreduce.am.env</name>
	<value>HADOOP_MAPRED_HOME=/usr/local/hadoop-3.0.0</value>
</property>
<property>
	<name>mapreduce.map.env</name>
	<value>HADOOP_MAPRED_HOME=/usr/local/hadoop-3.0.0</value>
</property>
<property>
	<name>mapreduce.reduce.env</name>
	<value>HADOOP_MAPRED_HOME=/usr/local/hadoop-3.0.0</value>
</property>
</configuration>
```

yarn-site.xml

```xml
<configuration>
<property>
	<name>yarn.nodemanager.aux-services</name>
	<value>mapreduce_shuffle</value>
</property>
<property>  
	<name>yarn.nodemanager.vmem-check-enabled</name>  
	<value>false</value>  
</property>  
</configuration>
```

## 1.4 格式化

执行格式化命令，以初始化HDFS。

```shell
hdfs namenode -format
```

当输出``succeessfully formatted``就表示格式化成功了。

## 1.5 运行Hadoop

```shell
sbin/start-all.sh
```

## 1.6 验证

1. 在终端输入jps命令，查看运行后台
2. 访问web端：
   1. localhost:9870
   2. localhost:8088

# 2. 安装Spark

## 2.1 安装Scala

Scala的安装和java的安装类似。解压，添加环境变量就行

```shell
tar -zxvf $SCALA_FILE -C $YOUR_OUTPUT_PATH
sudo vim /etc/profile
	# 添加
	export SCALA_HOME=/usr/local/scala-2.12.5
	export PATH=$PATH:$JAVA_HOME/bin:$SCALA_HOME/bin:$HADOOP_HOME/bin
source /etc/profile
```

执行完毕之后，可以在shell中输入``scala``，运行scala

## 2.2 安装Spark

解压spark，添加环境变量

```shell
tar -zxvf $SPARK_FILE -C $YOUR_OUTPUT_PATH
sudo vim /etc/profile
	# 添加
	export SPARK_HOME=/usr/local/spark-2.3.0-bin-hadoop2.7
	export PATH=$PATH:$JAVA_HOME/bin:$SCALA_HOME/bin:$HADOOP_HOME/bin:$SPARK_HOME/bin
source /etc/profile
```

修改spark文件夹下的conf文件夹下的配置文件。在spark-env.sh中添加

```shell
export JAVA_HOME=/usr/local/jdk1.8.0_162
export SCALA_HOME=/usr/local/scala-2.12.5
export HADOOP_HOME=/usr/local/hadoop-3.0.0
export HADOOP_CONF_DIR=/usr/local/hadoop-3.0.0/etc/hadoop
```

## 2.3 验证安装

1. 在终端中输入spark-shell
2. 在web端查看：localhost:8080

# 3. 安装Hive

## 3.1 安装Mysql

Hive内置了一个derby数据库来存储元数据。但是derby肯定没有mysql好用，而且mysql支持远程访问。所以这里先安装mysql，然后将mysql和hive整合。mysql在ubuntu下的安装很简单：

```shell
sudo apt-get install mysql-server mysql-client libmysqlclient-dev
```

安装过程中注意设置root用户的密码。安装完毕之后，输入``mysql -u root -p``进入mysql中。我们还需要设置mysql的远程访问：打开mysqld.cnf文件，注释掉bind-address=127.0.0.1

```shell
sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf
```

## 3.2 配置Hive

解压hive，添加环境变量：

```shell
tar -zxvf $HIVE_FILE -C $YOUR_OUTPUT_PATH
sudo vim /etc/profile
	# 添加
	export HIVE_HOME=/usr/local/apache-hive-2.3.2-bin
	export PATH=$PATH:$JAVA_HOME/bin:$SCALA_HOME/bin:$HADOOP_HOME/bin:$HIVE_HOME/bin:$SPARK_HOME/bin
source /etc/profile
```

为了建立mysql和hive的链接，还需要把mysql-connector-java.jar放在hive下的lib目录之中。然后修改conf文件夹下的配置文件：hive-site.xml (也就会hive-default.xml，注意重命名)和hive-env.sh。

添加下面的信息到hive-env.sh文件

```shell
# Set HADOOP_HOME to point to a specific hadoop install directory
export HADOOP_HOME=/usr/local/hadoop-3.0.0

# Hive Configuration Directory can be controlled by:
export HIVE_CONF_DIR=/usr/local/apache-hive-2.3.2-bin/conf

# Folder containing extra libraries required for hive compilation/execution can be controlled by:
export HIVE_AUX_JARS_PATH=/usr/local/apache-hive-2.3.2-bin/lib
```

添加mysql的信息。在hive-site.xml文件中的响应地方，修改为如下内容

```xml
<property>
　　<name>javax.jdo.option.ConnectionURL</name>
　　<value>jdbc:mysql://master:3306/hive?createDatabaseIfNotExist=true</value>
</property>
<property>
　　<name>javax.jdo.option.ConnectionDriverName</name>
　　<value>com.mysql.jdbc.Driver</value>
</property>
<property>
　　<name>javax.jdo.option.ConnectionUserName</name>
　　<value>root</value>
</property>
<property>
　　<name>javax.jdo.option.ConnectionPassword</name>
　　<value>1234</value>
</property>
```

然后将hive-site.xml文件中的``${system:java.io.tmpdir}``替换为响应的路径。比如我替换为``/usr/local/apache-hive-2.3.2-bin/tmp``。这里的tmp目录是我新建的。然后替换hive-site.xml文件中的``${system:user.name}``。这里我单纯的删掉了``system:``，所以替换为``${user.name}``。如果不进行替换，程序会报错：

```java
Exception in thread "main"java.lang.RuntimeException: 
java.lang.IllegalArgumentException:java.net.URISyntaxException: Relative path in absolute URI:${system:java.io.tmpdir%7D/$%7Bsystem:user.name%7D
```

## 3.3 验证安装

shell中执行``hive``，进入hive shell。



