# Process for installing software on a new VM
In the following document the ip to the newly created VM with Ubuntu is set to 1.1.1.1. It should also be set up with a public key login for user, and this user should be given sudo rights.


## Connect
> ssh user@1.1.1.1

> sudo apt-get install ssh git tmux 

> ssh-keygen -t rsa -C "me@me.com"


## Clone
> git clone git@github.com:digitalepidemiologylab/covid-bert.git

> cd covid-bert

> git submodule update --init

## Conda
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

> bash Miniconda3-latest-Linux-x86_64.sh

Answer “yes” to the last question and close and reopen connection.

> conda create -n python36 python=3.6

## Auto restart Conda environment (optional)
> echo "conda activate python36" >> ~/.bashrc

Close and reopen connection.

## Update environment
> cd covid-bert

> pip install -r requirements.txt

## Set login credentials Gcloud
> gcloud auth login

> gcloud auth application-default login 

## Attach tensorboard
> sh -N -f -L localhost:8881:localhost:6006 user@1.1.1.1

Open browser: http://localhost:8881. An alternative way of doing this is through the Google Console, and connect to the model-dirs in the bucket

## Update tf-nightly 
In this software we are running tf-nightly. In the future tensorflow 2.2 will most likely work. When using tf-nightly it is important that the VM and the TPUs are running the same versions. This command updates to latest version of tf-nighly on the VMs

> pip install tf-nightly --upgrade
