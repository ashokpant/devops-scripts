#!/bin/bash

#------------------------------------------------------------------------------
# Name:    sbtmkdirs
# Version: 1.5
# Purpose: Create an SBT project directory structure with a few simple options.
# Author:  Alvin Alexander, http://alvinalexander.com
# License: Creative Commons Attribution-ShareAlike 2.5 Generic
#          http://creativecommons.org/licenses/by-sa/2.5/
#------------------------------------------------------------------------------
declare -r TRUE=0
declare -r FALSE=1

# takes a string and returns true if it seems to represent "yes"
function isYes() {
  local x=$1
  [ $x = "y" ] && echo $TRUE; return
  [ $x = "Y" ] && echo $TRUE; return
  [ $x = "yes" ] && echo $TRUE; return
echo $FALSE
}

echo "This script creates an SBT project directory beneath the current directory."

while [ $TRUE ]; do

echo ""
  read -p "Directory/Project Name (MyFirstProject): " directoryName
  directoryName=${directoryName:-MyFirstProject}

  read -p "Create .gitignore File? (Y/n): " createGitignore
  createGitignore=${createGitignore:-y}

  read -p "Create README.md File? (Y/n): " createReadme
  createReadme=${createReadme:-y}

  echo ""
  echo "-----------------------------------------------"
  echo "Directory/Project Name: $directoryName"
  echo "Create .gitignore File?: $createGitignore"
  echo "Create README.md File?: $createReadme"
  echo "-----------------------------------------------"
  read -p "Create Project? (Y/n): " createProject
  createProject=${createProject:-y}
  [ "$(isYes $createProject)" = "$TRUE" ] && break

done

mkdir -p ${directoryName}/src/{main,test}/{java,resources,scala}
mkdir ${directoryName}/lib ${directoryName}/project ${directoryName}/target

# optional
#mkdir -p ${directoryName}/src/main/config
#mkdir -p ${directoryName}/src/{main,test}/{filters,assembly}
#mkdir -p ${directoryName}/src/site

#---------------------------------
# create an initial build.sbt file
#---------------------------------
echo "name := \"$directoryName\"

version := \"1.0\"

scalaVersion := \"2.12.8\"

libraryDependencies ++= Seq(
    \"org.scalatest\" %% \"scalatest\" % \"3.0.5\" % \"test\"
)

// see https://tpolecat.github.io/2017/04/25/scalac-flags.html for scalacOptions descriptions
scalacOptions ++= Seq(
    \"-deprecation\",     //emit warning and location for usages of deprecated APIs
    \"-unchecked\",       //enable additional warnings where generated code depends on assumptions
    \"-explaintypes\",    //explain type errors in more detail
    \"-Ywarn-dead-code\", //warn when dead code is identified
    \"-Xfatal-warnings\"  //fail the compilation if there are any warnings
)

" > ${directoryName}/build.sbt

#------------------------------
# create .gitignore, if desired
#------------------------------
if [ "$(isYes $createGitignore)" = "$TRUE" ]; then
echo "bin/
target/
build/
.cache
.cache-main
.classpath
.history
.project
.scala_dependencies
.settings
.worksheet
.DS_Store
*.class
*.log
*.iml
*.ipr
*.iws
.idea" > ${directoryName}/.gitignore
fi

#-----------------------------
# create README.me, if desired
#-----------------------------
if [ "$(isYes $createReadme)" = "$TRUE" ]; then
touch ${directoryName}/README.md
fi

echo ""
echo "Project created. See the following URL for build.sbt examples:"
echo "http://alvinalexander.com/scala/sbt-syntax-examples"
