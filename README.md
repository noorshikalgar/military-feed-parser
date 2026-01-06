# military-feed-parser

### Steps to use


#### 1. Clone the repository
```bash
git clone https://github.com/noorshikalgar/military-feed-parser.git
cd military-feed-parser
```

#### 2. Create venv, activate it and Install the requirements
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
This above can be like, python3 or py as you have installed it

#### 3. Run the program
```bash
# simple
python3 feed-parser-v2.py

# with options
# Full extraction (slower but complete content)
python3 feed-parser-v2.py --days 7

# Fast mode (only titles/descriptions)
python3 feed-parser-v2.py --no-extract --days 14

# Export to CSV/Excel
python3 feed-parser-v2.py --export csv

# Debug mode
python3 feed-parser-v2.py --debug --days 30

```


#### 4. serve the json file as an api via json-server
```bash
# -p is port -> which is set to 3400
json-server military_feed.json -p 3400
```