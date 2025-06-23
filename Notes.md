### tmux info
- list instances: tmux ls
- attach to instance: tmux a -t <instance>
- close tmux window: ctrl+b then d

### 26.06.25
- clarify which additional models we want to test -> how much additional disk space is required (disk currently full)

### 28.05.25

- Model selection -> try different sizes of models? Impact on performance (runtime, results)?
=> take a smaller and larger model of each LLM for comparison
- Evaluation -> needs rework
- Decision section does not cover last sentence which is annotated as an event! 
=> Include the sentence after the decision sections ends, plus the sentence before

- Preprocessing: add numbers to sentences to check order
- Length of documents: check context length of LLMs
- Check acceptable tokens of each LLM
- try out different prompts (very little to very structured)
- Make it more explicit to have the actual sentence as well in the output

### 25.04.25

- Langflow web editor
- 1. experiment: just dates
- Performance 
- Prompt engineering: how can we modify the prompt?
- Automatic evaluation of responses -> add them to the gate document and use gate for evaluation?
- Multi agent system: let the LLMs discuss with each other (like we did) and vote in the end

- second pass over results -> let LLM restructure it to make it more precise ("instruct" llm?)
=> IEbyLLM.ipynb
- alternative: https://python.useinstructor.com/
=> IEbyLLM_instructor.ipynb (currently not doing what it should do) 