Parallel Keyword Based News Content and Analysis
This project contributes a parallel, keyword-driven web crawling system that aggregates headline level and anchor-tag level news content from multiple Bangladeshi and English online newspapers.

Quick start

Install python3 from https://www.python.org/downloads/

Create a python virtual environmennt

python3 -m venv venv
Install dependencies:
python3 -m pip install -r requirements.txt
Run the project:
python3 main.py

Usage

Sequential:
`python main.py election --mode sequential`

Parallel:
`python main.py election --mode parallel --workers 4`

Benchmark:
`python main.py election --benchmark --workers 8`
