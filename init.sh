mkdir checkpoints

# Download SBU-shadow dataset and rename folder.
wget http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip
unzip SBU-shadow.zip -d ./
mv SBU-shadow data
rm SBU-shadow.zip

