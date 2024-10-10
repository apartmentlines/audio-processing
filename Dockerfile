FROM ghcr.io/jim60105/whisperx:large-v2-en

 # Switch to root user temporarily
 USER root

 # Update package list and install git
 RUN apt-get update && \
     apt-get install -y git

 # Switch back to the non-root user (assuming 1001 is the correct user ID)
 USER 1001

 # Set the working directory
 WORKDIR /app

# Change ownership of the /app directory to user 1001
# RUN chown -R 1001:1001 /app

# Set the user to run the container
USER 1001

# Set the working directory
WORKDIR /app
