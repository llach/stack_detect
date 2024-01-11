# ROS 2 - Python template

This repository is created as a starting template for ROS 2 projects using Python and provides two main things:

1. **CircleCI pipeline** for building and testing the ROS 2 package
2. **[Pre-commit hooks](https://pre-commit.com/)** for auto-formatting and linting

## How to use
When you click on the button to create a new repository, the very first option should allow you to choose the desired template. Alternatively, you can click on the **'Use this template'** button in the top-right of this page and select 'Create a new repository'.

### Restoring Submodules
Each component will have a dependency on the [common interfaces](https://github.com/softenable/ros2_common_interfaces). ROS messages, services and actions should be defined in common interfaces. These have been included as a submodule in the template. When you have cloned your repository ensure you update the submodule:
```git submodule update --init --recursive```

### Installing the pre-commit hooks locally
This template provides a configuration for the pre-commit hooks, however the pre-commit hooks **must be installed locally** for it to work

1. Run ```pre-commit install``` in the root of the repository to install the pre-commit hooks.
2. Install pylint using ```pip install pylint``` as it is required for one of the pre-commit hooks.

### Setting up CircleCI
When a new repository is created, it needs to be **set up through the CircleCI website**. To do this, follow the steps below:
1. Sign in to [CircleCI](https://app.circleci.com) with your GitHub account that has access to the organisation
2. Make sure SestoSenso is selected in the drop-down at the top left and click on 'Projects'.
4. Find the repository you just created and click on **'Set Up Project'**.
5. Select the 'Fastest' option and enter 'main' in the branch text box. Click on Set Up Project
6. After it sets up, go to 'Project Settings' -> Advanced, and enable **'Only build pull requests'**. This setting makes the workflow only run when either a pull request is created, or a commit is made to the default branch
7. The SSH keys for the repository are set up automatically. However, if a second git module or submodule is needed then an additional SSH key is required. For example, most repos will have a submodule: [ros2_common_msgs](https://github.com/softenable/ros2_common_interfaces). Navigate to the SSH Keys tab in the project settings in CircleCI. Add an additional SSH key with the hostname "github.com" and the value of the private SSH key for ros2_common_msgs can be copied from [here](https://kth-my.sharepoint.com/personal/fpokorny_ug_kth_se/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Ffpokorny%5Fug%5Fkth%5Fse%2FDocuments%2FSoftEnable%2F09%2E%20WPs%2FWP7%2FIntegration%2Fcommon%5Fmsgs%5Fcircle%5Fci%2Etxt&parent=%2Fpersonal%2Ffpokorny%5Fug%5Fkth%5Fse%2FDocuments%2FSoftEnable%2F09%2E%20WPs%2FWP7%2FIntegration
). If another submodule is required follow this [guide](https://circleci.com/docs/github-integration/#create-additional-github-ssh-keys). Remember to add the finger print to .circleci/config.yaml.

To view and configure a pipeline, you just have to be signed in to CircleCI with your GitHub account and you will be able to view all the details of the pipeline and edit the worfklow file from the CircleCI website as well.

## CircleCI pipeline
This template provides a CircleCI pipeline using the ```config.yml``` file in the ```.circleci``` folder that consists of the following steps:

1. Checkout repository and **install dependencies**
2. **Build ROS package** using ```colcon build```
3. **Execute unit tests** using ```colcon test```
4. Run ```pylint``` and ```pycodestyle``` to **lint all the files**

The ```config.yml``` file defines a ```ros-distro``` variable at the start of the file under the ```parameters``` section that contains the **value of the ROS 2 distro being used** (default: humble). Change this according to the distro being used.

The workflow will run every time a pull request is created (or pushed to main directly).

**However, this workflow only serves as a starting point** and can be edited and expanded as needed ([CircleCI documentation](https://circleci.com/docs/)). It also relies on you **adding appropriate unit tests** for your code.

To view or change the CircleCI pipeline, simply go to https://app.circleci.com, sign in with your GitHub account, and find the repository under the 'Projects' section (assuming the repository has been already been set up to use CircleCI).

## Pre-commit hooks
Pre-commit hooks are scripts that will run every time a commit is made, and are useful for identifying problems before pushing the code.
This template comes with pre-commit hooks configured to use [autopep8](https://pypi.org/project/autopep8/) and [pylint](https://pypi.org/project/pylint/) that will enforce the rules specified in the Integration Process Documentation.

Autopep8 will auto-format your code, while Pylint will check other rules such as naming conventions.

If **autopep8** finds any issues, it will **automatically reformat your code**, and the code will need to be committed again. If **pylint** finds any issues, the **commit will keep failing** until all the issues are fixed.

## Branch protection settings

Since the organization is on the free plan, branch protection settings cannot be enforced. However, it is good practice to **never push directly to main**, and always create a separate branch, especially for major changes. And when creating a pull request to merge into main, **have at least one other person review the changes**, and also let the CircleCI pipeline run and **ensure all stages pass before merging**.

## README template

Apart from this markdown file which can be deleted once a new repository is created, there is another markdown file called ```README-template.md``` which provides a structure for the README file for your repository.
