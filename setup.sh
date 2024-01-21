#!/usr/bin/env bash
. settings.sh

# Download and install CARLA
mkdir -p $CARLA910_ROOT
cd $CARLA910_ROOT

if [ ! -f "CARLA_0.9.10.1.tar.gz" ]; then
  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
  tar -xf CARLA_0.9.10.1.tar.gz
fi

if [ ! -f "AdditionalMaps_0.9.10.1.tar.gz" ]; then
  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
  tar -xf AdditionalMaps_0.9.10.1.tar.gz
fi

mkdir -p $CARLA911_ROOT
cd $CARLA911_ROOT

#if [ ! -f "CARLA_0.9.11.tar.gz" ]; then
#  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
#  tar -xf CARLA_0.9.11.tar.gz
#fi
#
#if [ ! -f "AdditionalMaps_0.9.11.tar.gz" ]; then
#  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.11.tar.gz
#  tar -xf AdditionalMaps_0.9.11.tar.gz
#fi
#
#mkdir -p $CARLA912_ROOT
#cd $CARLA912_ROOT
#
#if [ ! -f "CARLA_0.9.12.tar.gz" ]; then
#  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.12.tar.gz
#  tar -xf CARLA_0.9.12.tar.gz
#fi
#
#if [ ! -f "AdditionalMaps_0.9.12.tar.gz" ]; then
#  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.12.tar.gz
#  tar -xf AdditionalMaps_0.9.12.tar.gz
#fi
#
#mkdir -p $CARLA913a_ROOT
#cd $CARLA913a_ROOT
#
#if [ ! -f "CARLA_Leaderboard_20.tar.gz" ]; then
#  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/Leaderboard/CARLA_Leaderboard_20.tar.gz
#  tar -xf CARLA_Leaderboard_20.tar.gz
#fi

git submodule update --init --recursive
git submodule update --recursive --remote
