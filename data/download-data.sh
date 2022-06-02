if [[ -z $(which pip) ]]; then
	echo "You need pip installed to run this script"
	exit 1
fi
if [[ -z $(pip list | grep -F gdown) ]]; then
	delete_later=TRUE
	pip install gdown
fi	
echo "Downloading diamond data"
gdown 1rn5WY36ccx8PpHYy0WgvjEo6XFRuhUp_
gdown 1nQmHudPc65SxV5fovQie1ap7pCtOycuM 
[[ ! -z "$delete_later" ]] && pip uninstall gdown -y
