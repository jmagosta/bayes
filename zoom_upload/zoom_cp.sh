#/bin/bash
# Cron script to find zoom video files and copy them to Google Drive
# JMA Oct 2025

#Set debug mode for logging. 
set -x

# Address to alert that the new talk is up on Google Drive
TALK_EMAIL="talks@kannondo.org"

# Destination and Origin directories. 
GDRIVE_DIR="/Users/$USER/Library/CloudStorage/GoogleDrive-johnmark.agosta@gmail.com/My Drive"
ZOOM_DIR="/Users/$USER/Documents/Zoom"
LOG_DIR="/Users/$USER/Documents/Zoom_logs"       # TODO make sure this directory exists

# Ask the user for the name with which to save this session
read -p "Name to give this talk: " TALK_LABEL

# Switch to where the recordings are found
pushd $ZOOM_DIR > /dev/null

# Wait until Zoom is finished creating the recordings file
while true; do
  sleep 10
  # Does the new directory exist?
  if [[ ! condition ]]; then
    break
  fi
done

# Zoom recordings are in directories named with a timestamp preface, e.g. "2024-04-23 17.03.04 ..."
# Find most recent recording directory - just list chronologically
# The audio files look like "audio1468673967.m4a"
RECENT_DIR=$(ls -t | head -1)

# Each session consists of three files, m4a audio, mp4 video and a conf text file
echo "$ZOOM_DIR/$RECENT_DIR"

# File to copy
cd RECENT_DIR
AUDIO_FILE=$(ls *.m4a)

# Copy desired file to Google Drive
# TODO Create a new directory for it?
cp -R "${ZOOM_DIR}/${RECENT_DIR}/${AUDIO_FILE}" "${GDRIVE_DIR}"

# Rename file 
#TODO customize date format
NEW_FILE="$(DATE -I)_${TALK_LABEL}.m4a"
mv ${AUDIO_FILE} ${NEW_FILE}
# Send mail notification of transfer & log activity.  
echo "Find the most recent raw Zoom audio in ${NEW_FILE} "| mail -s "Zoom transfer on ${date}" "${TALK_EMAIL}"

# Log the script actions using ./zoom_cp.sh > logfile.txt

# Restore directory state
popd > /dev/null
