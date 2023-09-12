if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* || "$OSTYPE" == "msys" ]]; then
    pip install -r requirements.txt
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Dependencies installed successfully."
