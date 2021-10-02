#!/bin/sh

echo "📝 Setting up workspace project & hooks ..."

DIR=./venv
if [ -d "$DIR" ]; then
    echo "virtual env:$DIR exists."
else 
    echo "virtual env:$DIR does not exists... Creating new one!"
    python3 -m virtualenv venv 
    source venv/bin/activate
    pip3 install -r requirements.txt 
fi

sh ./scripts/hooks/setup.sh
echo "✅ Done setup project & hooks ...♥️ "