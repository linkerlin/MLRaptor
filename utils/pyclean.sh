# Recursively delete all *.pyc files in the repository
# NOTE: These are UNTRACKED by git anyways, so this is mostly for aesthetics
#  but can be useful when moving from 32bit to 64bit, changing virtenvs, etc.

pushd ~/git/MLRaptor/

find . -name "*.pyc" -delete

popd
