
# Table of Contents

1.  [Exploration](#exploration)
    1.  [General Guides](#guides)
    2.  [Prompt examples](#prompt-examples)
2.  [Evaluation](#evaluation)
    1.  [Datasets](#datasets)
    2.  [OpenAI evals](#OpenAI-evals)
3.  [Beyond Prompt Engineering](#beyond-prompt-engineering)

"The hottest new programming language is English" - Andrej Karpathy,
[24 Jan 2023](https://twitter.com/karpathy/status/1617979122625712128)

Prompt engineering is about skillfully creating input queries
(prompts) to communicate with AI models like ChatGPT
effectively. Think of it as writing instructions for a highly capable
yet sometimes unpredictably dumb personal assistant.

This guide serves as a hands-on resource for developers and early
adopters using large language models (LLMs). It goes beyond the usual
one-off task prompts, focusing instead on processing large quantities
of inputs via an API. When manual review of every output isn't
feasible, it's critical to evaluate and manage the trade-offs between
cost, speed, and output quality. Therefore we emphasize the
'engineering' part of prompt engineering here.

Our aim with this guide is to organize links to key external resources,
and give concise commentary to help you find what's relevant for your task.

If you want to contribute to this guide, please open an issue, send a
PR, or email me at prompts@matthiasberth.com.


<a id="exploration"></a>

# Exploration

In the exploration phase of prompt engineering, the focus is on
generating a range of candidate prompts that perform effectively on
example inputs. This phase involves using a playground environment to
experiment with various combinations of instructions, examples, and
inputs, allowing for the identification and resolution of
issues. Rapid iteration and drawing inspiration from existing prompts
in the wild are key strategies during this phase.


<a id="guides"></a>

## General Guides

1.  Prompt engineering guide from OpenAI

    The [Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering) guide from OpenAI covers "Six strategies for
    getting better results":
    
    -   [Write clear instructions](https://platform.openai.com/docs/guides/prompt-engineering/write-clear-instructions)
    -   [Provide reference text](https://platform.openai.com/docs/guides/prompt-engineering/provide-reference-text)
    -   [Split complex tasks into simpler subtasks](https://platform.openai.com/docs/guides/prompt-engineering/split-complex-tasks-into-simpler-subtasks)
    -   [Give the model time to "think"](https://platform.openai.com/docs/guides/prompt-engineering/give-the-model-time-to-think)
    -   [Use external tools](https://platform.openai.com/docs/guides/prompt-engineering/use-external-tools)
    -   [Test changes systematically](https://platform.openai.com/docs/guides/prompt-engineering/test-changes-systematically)
    
    Each comes with a set of tactics (like "Ask the model if it missed
    anything on previous passes"). The guide provides direct links to the
    OpenAI playground where you can try out examples.

2.  Microsoft [Introduction to prompt engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering),

    The intro covers common techniques and best practices. The [techniques](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions)
    article discusses Chain of Thought prompting, and the influence of the
    temperature parameter, among others.

3.  [Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4](https://arxiv.org/pdf/2312.16171.pdf)

    This research paper presents 26 guiding principles and evaluates their
    effectiveness across several models.

4.  The [CO-STAR framework](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41)

    Suggests to structure the prompt as Context, Objective, Style, Tone,
    Audience, Response. Makes a lot of sense and helped the author win a
    competition. I'm still [trying to track down](https://twitter.com/mberth/status/1745368013459603627) original sources to the
    CO-STAR framework and that competition.


<a id="prompt-examples"></a>

## Prompt examples

1.  [OpenAI prompt examples](https://platform.openai.com/examples)

    Many of these are geared to everyday use, but there are relevant
    prompts in the categories:
    
    -   [Extract](https://platform.openai.com/examples?category=extract), e.g. Classify user reviews based on a set of tags.
    -   [Transform](https://platform.openai.com/examples?category=transform), e.g. Convert ungrammatical statements into standard English.

2.  Prompt collections / Libraries

    -   [LangChain Hub](https://blog.langchain.dev/langchain-prompt-hub/) collects prompts in a variety of areas, e.g. Tagging,
    
    Summarization, Extraction.
    
    -   LangChain has prompts baked into its code. For example, here is a
        set of prompts for checking the correctness of summarizations:
        [langchain/chains/llm\_summarization\_checker/prompts](https://github.com/langchain-ai/langchain/tree/611f18c944cd8c17a70d8d6c89508c16d5856846/libs/langchain/langchain/chains/llm_summarization_checker/prompts). So you can look
        up a [use case](https://python.langchain.com/docs/use_cases) in the LangChain docs ([Summarization](https://python.langchain.com/docs/use_cases/summarization)) and locate the
        relevant code.
    
    e.g. langchain, Llamaindex

3.  Finding examples by tasks / use case

    Know the general category for your task, so you can search
    effectively for prompt examples, papers, and benchmark datasets.
    
    1.  Data Extraction
    
        Example: Extract product number, due date from unstructured orders received via email.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+data+extraction)
    
    2.  Sentiment Analysis
    
        Example: Analyzing customer feedback to determine sentiment towards a product or service.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+sentiment+analysis)
    
    3.  Chatbot Conversations
    
        Example: Developing chatbots for handling customer service inquiries.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+chatbot+conversations)
    
    4.  Text Classification
    
        Example: Categorizing support tickets into departments like technical, billing, general inquiries.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+text+classification)
    
    5.  Named Entity Recognition (NER)
    
        Example: Identifying company names in financial reports.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+named+entity+recognition)
    
    6.  Keyword Extraction
    
        Example: Extracting relevant keywords for SEO or document summarization.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+keyword+extraction)
    
    7.  Language Translation
    
        Example: Translating business documents or communications between languages.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+language+translation)
    
    8.  Summarization
    
        Example: Generating concise summaries of long documents like business reports.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+summarization)
    
    9.  Topic Modeling
    
        Example: Identifying main topics in customer feedback or a collection of articles.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+topic+modeling)
    
    10. Spam Detection
    
        Example: Filtering out spam comments in a forum.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+spam+detection)
    
    11. Intent Recognition
    
        Example: Understanding the intent behind customer messages in chatbot interactions.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+intent+recognition)
    
    12. Text Generation
    
        Example: Automatically generating text like product descriptions based on data inputs.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+text+generation)
    
    13. Question Answering Systems
    
        Example: Building systems for answering customer questions in natural language.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+question+answering+systems)
    
    14. Emotion Detection
    
        Example: Identifying emotional states in text to understand customer sentiment.
        [(google this)](https://www.google.com/search?q=prompt+engineering+for+emotion+detection)


<a id="evaluation"></a>

# Evaluation


<a id="datasets"></a>

## Datasets

1.  [Papers with code datasets](https://paperswithcode.com/datasets)

2.  [HuggingFace datasets](https://huggingface.co/docs/datasets/index)

3.  [Kaggle Datasets - NLP](https://www.kaggle.com/datasets?tags=13204-NLP)


<a id="OpenAI-evals"></a>

## [OpenAI evals](https://github.com/openai/evals)

Evals is a framework for evaluating LLMs and LLM systems, and an
open-source registry of benchmarks.


<a id="beyond-prompt-engineering"></a>

# Beyond Prompt Engineering

When all your prompt engineering efforts don't give good enough
results, you can try some alternatives

-   Use another model. If you haven't done so already, try a different
    model with roughly the same or better capabilities. Keep in mind
    that performance is determined by the combination of model and
    prompt so you may want to iterate on your best prompt.
-   Fine-tune an existing model. You can select examples from your
    current dataset, or create them by hand.
-   Invest in better examples for a few-shot prompt. Think about
    providing more examples, more diverse examples, and positive
    vs. negative examples. If you're using RAG, try investing in the
    retrieval part of the pipeline.
-   Use ensembles / mixture of experts. Solve the same task by multiple
    different prompts / models, then consolidate results with a majority
    vote or some other mechanism.
-   Use automated methods to find a better prompt and / or better
    examples. For example, the [DSPy paper](https://arxiv.org/abs/2310.03714) reports performance
    improvements of 16-40% for their auto-optimized pipelines.
-   Roll your own NLP solution. For some tasks, you don't necessarily
    need the large language model, it's just much more convenient to
    use. There is a wide array of more classical NLP methods that you
    may want to use. You can still let LLMs help you with generating
    enough labeled data.
-   Pause. Seriously, sometimes it may be a viable approach to move on to
    the next promising application of LLMs. While you do that, something
    new may come up, like a price drop, a new more advanced model, or
    some research breakthrough that makes it worth revisiting the task.

