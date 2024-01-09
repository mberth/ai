# Prompt Engineering Resources


"The hottest new programming language is English" - Andrej Karpathy, [24 Jan 2023](https://twitter.com/karpathy/status/1617979122625712128)

Prompt engineering is the process of carefully crafting input queries
(prompts) to effectively communicate with AI models, like ChatGPT, to
get desired or more accurate outputs. It's a bit like learning how to
write instructions for a smart personal assistant that is dumb in
unexpected ways.

In contrast to more casual use of a chat model, prompt *engineering*
focuses on repeatable results that have 

-   good quality
-   low cost
-   good safety
-   good throughput
-   low effort in developing the whole system

As usual in engineering, this is about tradeoffs, e.g. you can put more
effort into crafting a prompt that allows you to use a cheaper model.


<a id="org622f627"></a>

# General Guides


<a id="org5ae6954"></a>

## Prompt engineering guide from OpenAI

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


<a id="org86b8f2e"></a>

## Microsoft

[Introduction to prompt engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
<https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions>


<a id="orgcd8a3ee"></a>

# Prompt examples


<a id="org3ed634d"></a>

## [OpenAI prompt examples](https://platform.openai.com/examples)

Many of these are geared to everyday use, but there are relevant
prompts in the categories:

-   [Extract](https://platform.openai.com/examples?category=extract), e.g. Classify user reviews based on a set of tags.
-   [Transform](https://platform.openai.com/examples?category=transform), e.g. Convert ungrammatical statements into standard English.


<a id="orgb16576d"></a>

## Prompt collections / Libraries

e.g. langchain, Llamaindex


# Bag of tricks



# Structured responses


<a id="orgb8e4e06"></a>

## Guidance, LMQL, RELM, and Outlines

> are all exciting new libraries for controlling the individual
completions of LMs, e.g., if you want to enforce JSON output schema or
constrain sampling to a particular regular expression.


<a id="org3424e99"></a>

# Advanced Prompt Engineering


<a id="orgdf9c0d4"></a>

## CoT etc


<a id="org14638e0"></a>

## Automated prompt engineering


<a id="orgb65c930"></a>

## Stanford: DSPy


<a id="orgd8da6af"></a>

## Promptroyale


<a id="org12b9f7d"></a>

## Microsoft: medprompt

<https://github.com/microsoft/promptbase/tree/main>


<a id="org7a3dde7"></a>

# Evaluation


<a id="org30478b8"></a>

## andere: evals von openAI


<a id="org9b66d46"></a>

## [OpenAI evals](https://github.com/openai/evals)  Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.


<a id="org94cec2f"></a>

## [Evals framework von OpenAI](https://github.com/openai/evals)


<a id="org621db2f"></a>

# Tools

You can also run and create evals using Weights & Biases.

A/B testing stats übertragen


<a id="org99a8610"></a>

# Risks


<a id="orgadaf11c"></a>

## What can go wrong?


<a id="org149f599"></a>

## Social enginering

<https://media.ccc.de/v/37c3-12008-unsere_worte_sind_unsere_waffen#t=601>

prompt injection, fight of the system prompt vs user prompt / chat
input

Eva Wolfangel


<a id="org7dfda5a"></a>

## Extract training data

<https://arxiv.org/pdf/2311.17035.pdf>
[Scalable Extraction of Training Data from (Production) Language Models](https://arxiv.org/pdf/2311.17035.pdf)

This paper studies extractable memorization: training data
that an adversary can efficiently extract by querying a ma-
chine learning model without prior knowledge of the training
dataset. We show an adversary can extract gigabytes of train-
ing data from open-source language models like Pythia or
GPT-Neo, semi-open models like LLaMA or Falcon, and
closed models like ChatGPT


<a id="orgafef27c"></a>

## Prompt Injection

-   Abuse a system (chat chevrolet?)
-   Access private data
-   Perform actions in the user's name


<a id="org1d58e88"></a>

## Prompt Extraction

-   Understand how a system works
-   Copy a system
-   Prepare for prompt injection


<a id="orgaa6399e"></a>

## Recommendation

If it's uncontrolled input, then:

-   Only use prompts that you could publish (Simon Wilison: publish your prompts)
-   Do not process sensitive data
-   Be cautious with tools and code execution. Every website is a potential tool if HTTP get requests
    are allowed. Code execution allows creative workarounds for any
    safeguards you may have in your prompt.


<a id="orga9ab31a"></a>

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

