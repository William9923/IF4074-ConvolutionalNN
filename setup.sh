#!/bin/sh

echo "📝 Setting up workspace project & hooks ..."
python3 -m virtualenv venv 
source venv/bin/activate

pip3 install -r requirements.txt 

sh ./scripts/hooks/setup.sh
echo "✅ Done setup project & hooks ...♥️ "