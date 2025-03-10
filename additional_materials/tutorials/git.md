# Git Basics

## Homework 

To prepare for the course and ensure you have the necessary tools and knowledge, please complete the following tasks:

### 1. Install Git

- Follow the instructions to install Git on your system: [Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### 2. Create a GitHub Account and add SSH key

- Sign up for an account on [GitHub](https://github.com/) if you do not already have one.
- Create private/public ssh keys: [link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=mac) (at the top of the page select OS you are using)
- Add your public SSH key to your github account: [link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?tool=webui&platform=mac) (at the top of the page select OS you are using)

### 3. (Optional) Integrate GitHub with PyCharm

- Configure PyCharm to connect to your GitHub account. Detailed steps can be found here: [PyCharm GitHub Integration](https://www.jetbrains.com/help/pycharm/github.html).

### 4. Learn Git Basics

- Familiarize yourself with basic Git commands and concepts by reading Chapters 2.1 – 2.4 on this resource: [Git Handbook](https://fri-datascience.github.io/course_ids/handbook/git.html).

## Exercise 1: Local Repositories

In this exercise, you'll practice working with local repositories:

1. **Create a New Project in PyCharm**:
    - Start by creating a new project in PyCharm, ensuring you include a Python virtual environment.

2. **Initialize Git**:
    - Within your new project, initialize a Git repository.

3. **Practice Basic Git Commands**:
    - Get hands-on experience with the following Git commands:
        - `commit`: Save changes to your local repository.
        - `push`: Upload your committed changes to a remote repository.
        - `pull`: Fetch and merge changes from the remote repository to your local repository.
        - `checkout`: Switch between branches or revert files to a previous state.
        - `diff`: Compare different versions of files.
        - `gitignore`: Specify intentionally untracked files to ignore.
        - `reset`: Undo changes by resetting the current HEAD to a previous state.
        - `merge`: Combine changes from different branches.
        - `clone`: Copy a remote repository to your local machine.

## Exercise 2: Remote Repositories

This exercise focuses on collaborating using remote repositories:

1. **Create Your Repo on GitHub**:
    - Create it from scratch on Github.
    - To upload an existing local repo, you need to install [Github CLI](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github#adding-a-local-repository-to-github-with-github-cli)
   
2. **Collaborate on a Shared Project**:
    - Clone the following repository to start collaborating: [Git Introduction Repository](https://github.com/azagsam/git-intro).

3. **Add Collaborators**:
    - Learn how to add members to your GitHub project, allowing them to contribute.

4. **Engage in Collaboration**:
    - Change code, commit your changes, push them to the remote repository, and update your local project with changes made by others. Practice resolving conflicts that arise when multiple collaborators edit the same piece of code.
