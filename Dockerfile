FROM ghcr.io/jim60105/whisperx:large-v2-en

# Change ownership of the /app directory to user 1001
# RUN chown -R 1001:1001 /app

# Set the user to run the container
USER 1001

# Set the working directory
WORKDIR /app
