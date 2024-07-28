# Gen Slide
Generating PowerPoint slides extracted from a text.

## How to run

### Set up 
Clone the repo: 
```commandline
clone git http://github.com/mehdimo/GenSlide

cd GenSlide
```

then
```commandline
python -m venv ./venv
. ./venv/bin/activate

pip install -r requirements.txt
```
* Note: Use a Python != 3.9.7 for virtualenv. My version is 3.12. 
* To learn more about gpt4all, see [here](https://docs.gpt4all.io/).

### RUN LLM Service
1. Go to `llm-service` folder and run the `gpt.py` file.
```commandline
cd llm-service
python gpt.py
```
* Running for the first time, the LLM model will be downloaded which may take several minutes.

### RUN UI
Navigate to `fronend` folder and run `ui.py` using streamlit command:
```commandline
cd .. 
cd frontend
streamlit run ui.py
```
It will open the UI in the browser. 

