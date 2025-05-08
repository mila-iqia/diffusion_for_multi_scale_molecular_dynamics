# Using FLARE


The FLARE model leverages Gaussian processes to predict interatomic energies and forces.
The code base is available on github: https://github.com/mir-group/flare.

It is challenging to install FLARE as it relies on compiling C++ code and LAMMPS. To 
simplify installation, we describe here how to use a container to compile everything
in a well-defined environment.


# Container 
We use [Podman](https://podman.io/) for creating and using containers. This is very similar
to Docker, but open-source.

Although the podman ecosystem is "serveless", a virtual machine is needed on mac.
This can be obtained from homebrew. Podman Desktop is also very useful and can be 
obtained from homebrew.

Note that in order to be able to create containers, the virtual machine must be started.
Be careful to specify enough memory! The default is 2G, which is not enough to compile FLARE.

    podman machine init --cpus 4 --memory 16384
    podman machine start

## Commands
The file `ContainerFile` contains the "recipe" to create an image with FLARE installed and 
a running ssh server. The latter is very useful to connect an IDE (say, PyCharm) to explore
the running code inside a container.

Note that FLARE is intended to run on x86_64 architectures (which can be emulated on a mac). 
We will explicitly specify this architecture in what follows.

## Create the image
To create the image, create a directory (say, `build_podman`) and put the `ContainerFile` in it.
The image can then be created with the following command.

    podman build --arch x86_64  -t flare -f ./build_podman/ContainerFile

This takes quite a bit of time.

## Start a container from the image
Once the image is built, a container can be started with

    podman run -d --arch x86_64 -p8022:22 flare 

This command creates a container in "detached" mode (i.e., in the background). This starts the internal ssh server,
which exposes port 22.

In order to connect to the container by ssh, we must copy the *ssh private key* from inside the container
to outside of it.

    podman cp <container_id>:/home/flare_user/.ssh/id_rsa ./id_rsa
    chmod 600 ./id_rsa

The `container_id` can be obtained using the command

    podman ps

It should now be possible to ssh into the running container with

    ssh flare_user@localhost -p 8022 -i ./id_rsa

Careful! When connecting to the container is this way, an entry is added in the `$HOME/.ssh/known_hosts` file. If 
you destroy the container and create it again, this entry will no longer match the new ssh server created, and 
ssh will complain that something is wrong: simply delete the offending lines in `$HOME/.ssh/known_hosts`.


Once sshed into the container, the python tests can be executed by

    # The tests depend on this environment variable pointing to the lammps executable.
    export lmp=/home/flare_user/flare/lammps/build/lmp 
    cd flare/
    source .venv/bin/activate
    cd tests/ 
    pytest

# Stop a container
To stop the container, simply use
    
    podman stop <container_id>

Containers can also be stopped or deleted using the `Podman Desktop` App (a GUI).

# Running a Jupyter-Notebook 
To run a Jupyter-Notebook that lives inside the container but that is displayed on a browser outside,
do the following.

1. OUTSIDE THE CONTAINER: launch a container and forward ports for ssh and the browser:

        podman run -d --arch x86_64 -p8022:22 -p8888:8888 flare
 
   2. INSIDE THE CONTAINER: launch the notebook (must be in the right folder)
      - ssh into the container  
      - source the python environment (source flare/.venv/bin/activate)
      - install `jupyter` (uv install jupyter)
      - start the Jupyter notebook in the `tutorials/` folder.

               jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root sparse_gp_tutorial.ipynb

       Launching the notebook should produce instructions on how to connect. Something like

            To access the server, open this file in a browser:
            file:///home/flare_user/.local/share/jupyter/runtime/jpserver-672-open.html
            Or copy and paste one of these URLs:
                http://19db8729c0f1:8888/tree?token=efb7dd023988a8324359edb99e66db596fe97ae0ff827d24
                http://127.0.0.1:8888/tree?token=efb7dd023988a8324359edb99e66db596fe97ae0ff827d24

        Simply copy-paste the http link (ie, "http://127.0.0.1:8888/...") and use in a browser. That
        should show the notebook!

# Other common podman commands

    podman ps  # see what containers are running
    podman images # see what images are present
    podman rmi <image id> # delete an image
    podman run --arch x86_64 -it flare /bin/bash  # start a container in interactive mode

