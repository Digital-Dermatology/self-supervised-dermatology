.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := self_supervised_dermatology
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_BRANCH := $(if $(GIT_BRANCH),$(GIT_BRANCH),main)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)

# docker
GPU_ID := 0

# check if `netstat` is installed
ifeq (, $(shell which netstat))
$(error "Netstat executable not found, install it with `apt-get install net-tools`")
endif

# Check if Jupyter Port is already use and define an alternative
ifeq ($(origin PORT), undefined)
  PORT_USED = $(shell netstat -tln | grep -E '(tcp|tcp6)' | grep -Eo '8888' | tail -n 1)
  # Will fail if both ports 9999 and 10000 are used, I am sorry for that
  NEXT_TCP_PORT = $(shell netstat -tln | grep -E '(tcp|tcp6)' | grep -Eo '[0-9]{4}' | sort | tail -n 1 | xargs -I '{}' expr {} + 1)
  ifeq ($(PORT_USED), 8888)
    PORT = $(NEXT_TCP_PORT)
  else
    PORT = 8888
  endif
endif

DOCKER_CMD := docker run --rm -u $(id -u):$(id -g) -v $$PWD:/workspace/ -v /media/gengar/data-repository/:/media/gengar/data-repository/ --name $(PROJECTNAME)_no_gpu --shm-size 8G -it $(PROJECTNAME):$(GIT_BRANCH)
DOCKER_GPU_CMD := docker run --rm -p $(PORT):8888 -u $(id -u):$(id -g) -v $$PWD:/workspace/ -v /media/gengar/data-repository/:/media/gengar/data-repository/ --gpus='"device=$(GPU_ID)"' --name $(PROJECTNAME)_gpu_$(GPU_ID) --shm-size 40G -it $(PROJECTNAME):$(GIT_BRANCH)
DOCKER_DGX := docker run \
			  -it \
              -u $(id -u):$(id -g) \
			  -v ${PWD}:/workspace/ \
			  -v /raid/dataset/:/raid/dataset/ \
			  -w /workspace \
			  -d \
			  --gpus='"device=0,1,2,3"' \
              --name $(PROJECTNAME)_multi_gpu \
			  --shm-size 200G \
			  --env-file .env

TORCH_CMD := torchrun --nnodes 1 --node_rank 0 --nproc_per_node 4
ENV := prod
RUNCMD := BUILD_ENV=$(ENV) docker-compose up --build
# SSH
PORT := 22
USERNAME := fgroger
DEST_FOLDER := self-supervised-dermatology

###########################
# COMMANDS
###########################
# Thanks to: https://stackoverflow.com/a/10858332
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

###########################
# PROJECT UTILS
###########################
.PHONY: init
init:  ##@Utils initializes the project and pulls all the nessecary data
	@git submodule update --init --recursive

.PHONY: update_data_ref
update_data_ref:  ##@Utils updates the reference to the submodule to its latest commit
	@git submodule update --remote --merge

.PHONY: clean
clean:  ##@Utils cleanes the project
	@black .
	@nb-clean clean notebooks/*.ipynb
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f .coverage
	@rm -f core

.PHONY: test
test:  ##@Utils runs all tests in the project (unit and integration tests)
	docker-compose up --build run_tests

###########################
# SSH UTILS
###########################
.PHONY: push_ssh
push_ssh: clean  ##@SSH pushes all the directories along with the files to a remote SSH server
	$(call check_defined, SSH_CONN)
	rsync -r --exclude='.git/' --exclude='.github/' --exclude='wandb/' --progress -e 'ssh -p $(PORT)' $(PROJECT_DIR)/ $(USERNAME)@$(SSH_CONN):$(DEST_FOLDER)/

.PHONY: pull_ssh
pull_ssh:  ##@SSH pulls directories from a remote SSH server
	$(call check_defined, SSH_CONN)
	scp -r -P $(PORT) $(USERNAME)@$(SSH_CONN):$(DEST_FOLDER) .

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker runs an interacitve bash inside the docker image
	@echo "Run inside docker image"
	$(DOCKER_CMD) /bin/bash

run_gpu_bash: _build  ##@Docker runs an interacitve bash inside the docker image with a GPU
	@echo "Run inside docker image"
	$(DOCKER_GPU_CMD) /bin/bash

start_jupyter: _build  ##@Docker start a jupyter notebook inside the docker image (default: GPU=true)
	@echo "Starting jupyter notebook"
	@-docker rm $(PROJECTNAME)_gpu_$(GPU_ID)
	$(DOCKER_GPU_CMD) /bin/bash -c "jupyter notebook --allow-root --ip 0.0.0.0 --port 8888"

###########################
# TRAINING
###########################
train_simclr: ##@Training trains SimCLR
	$(RUNCMD) train_simclr

train_byol: ##@Training trains BYOL
	$(RUNCMD) train_byol

train_colorme: ##@Training trains ColorMe
	$(RUNCMD) train_colorme

train_dino: ##@Training trains DINO
	$(RUNCMD) train_dino

train_ibot: ##@Training trains iBOT
	$(RUNCMD) train_ibot

###########################
# TRAINING (MULTI GPU - DGX)
###########################
train_simclr_docker: _build  ##@Training-DGX trains SimCLR for Multi GPU training in Docker Container
	@$(eval ARCH = simclr)
	@-docker rm $(PROJECTNAME)_$(ARCH)
	@$(DOCKER_DGX) \
		--name $(PROJECTNAME)_$(ARCH) \
		$(PROJECTNAME):$(GIT_BRANCH) \
		$(TORCH_CMD) src/train_$(ARCH).py --config_path arguments/gpu/$(ARCH).yaml;
	@docker attach $(PROJECTNAME)_$(ARCH)

train_byol_docker: _build  ##@Training-DGX trains BYOL for Multi GPU training in Docker Container
	@$(eval ARCH = byol)
	@-docker rm $(PROJECTNAME)_$(ARCH)
	@$(DOCKER_DGX) \
		--name $(PROJECTNAME)_$(ARCH) \
		$(PROJECTNAME):$(GIT_BRANCH) \
		$(TORCH_CMD) src/train_$(ARCH).py --config_path arguments/gpu/$(ARCH).yaml;
	@docker attach $(PROJECTNAME)_$(ARCH)

train_colorme_docker: _build  ##@Training-DGX trains ColorMe for Multi GPU training in Docker Container
	@$(eval ARCH = colorme)
	@-docker rm $(PROJECTNAME)_$(ARCH)
	@$(DOCKER_DGX) \
		--name $(PROJECTNAME)_$(ARCH) \
		$(PROJECTNAME):$(GIT_BRANCH) \
		$(TORCH_CMD) src/train_$(ARCH).py --config_path arguments/gpu/$(ARCH).yaml;
	@docker attach $(PROJECTNAME)_$(ARCH)

train_dino_docker: _build  ##@Training-DGX trains DINO for Multi GPU training in Docker Container
	@$(eval ARCH = dino)
	@-docker rm $(PROJECTNAME)_$(ARCH)
	@$(DOCKER_DGX) \
		--name $(PROJECTNAME)_$(ARCH) \
		$(PROJECTNAME):$(GIT_BRANCH) \
		$(TORCH_CMD) src/train_$(ARCH).py --config_path arguments/gpu/$(ARCH).yaml;
	@docker attach $(PROJECTNAME)_$(ARCH)

train_ibot_docker: _build  ##@Training-DGX trains iBOT for Multi GPU training in Docker Container
	@$(eval ARCH = ibot)
	@-docker rm $(PROJECTNAME)_$(ARCH)
	@$(DOCKER_DGX) \
		--name $(PROJECTNAME)_$(ARCH) \
		$(PROJECTNAME):$(GIT_BRANCH) \
		$(TORCH_CMD) src/train_$(ARCH).py --config_path arguments/gpu/$(ARCH).yaml;
	@docker attach $(PROJECTNAME)_$(ARCH)

train_ibot_swin_docker: _build  ##@Training-DGX trains iBOT w/ Swin Transformer for Multi GPU training in Docker Container
	@$(eval ARCH = ibot)
	@-docker rm $(PROJECTNAME)_$(ARCH)
	@$(DOCKER_DGX) \
		--name $(PROJECTNAME)_$(ARCH) \
		$(PROJECTNAME):$(GIT_BRANCH) \
		$(TORCH_CMD) src/train_$(ARCH).py --config_path arguments/gpu/$(ARCH)_swin.yaml;
	@docker attach $(PROJECTNAME)_$(ARCH)

train_mae_docker: _build  ##@Training-DGX trains MAE for Multi GPU training in Docker Container
	@$(eval ARCH = mae)
	@-docker rm $(PROJECTNAME)_$(ARCH)
	@$(DOCKER_DGX) \
		--name $(PROJECTNAME)_$(ARCH) \
		$(PROJECTNAME):$(GIT_BRANCH) \
		$(TORCH_CMD) src/train_$(ARCH).py --config_path arguments/gpu/$(ARCH).yaml;
	@docker attach $(PROJECTNAME)_$(ARCH)

###########################
# EVAL
###########################
eval_imagenet: ##@Eval evaluates a pretrained ImageNet ResNet on downstream
	$(RUNCMD) eval_imagenet

eval_simclr: ##@Eval evaluates SimCLR on downstream
	$(RUNCMD) eval_simclr

eval_byol: ##@Eval evaluates BYOL on downstream
	$(RUNCMD) eval_byol

eval_dino: ##@Eval evaluates DINO on downstream
	$(RUNCMD) eval_dino

eval_ibot: ##@Eval evaluates IBOT on downstream
	$(RUNCMD) eval_ibot

eval_colorme: ##@Eval evaluates ColorMe on downstream
	$(RUNCMD) eval_colorme
