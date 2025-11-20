# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and install
rm -rf build dist
rm -rf *.egg-info
python3 setup.py bdist_wheel
pip3 install dist/*.whl --force-reinstall --no-deps

# Open users' original directory
cd "$original_dir"
