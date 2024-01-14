<div align="center">

[![logo](https://raw.githubusercontent.com/xmaayy/tinyvec/master/static/logo.png)](https://github.com/xmaayy/tinyvec)
tinyvec | A toy implementation with aspirations of being useful

</div>

---

An implemetation of a vector database whose purpose is not to serve you as fast as possible, but rather allow you to experiment without worrying if you have enough memory.  

## Features
### On Disk Vector Storage
Will this beat some fully hosted blazingly fast tensor library? Not likely. But when looking to do RAG on a set of documents I found that either II needed to set up some behemoth database, or I needed to have >60GB of memory (and occasionally both). I decided to see if it was possible to create something that was more aligned with the average user having a lot more disk space than available memory.

### LangChain compatability
This project was originally formed because I wanted to run a lot of medical papers through TinyLlama on CoLab. They only give you 12GB of memory, but about 100GB of disk which I'm willing to bet is an SSD. So I spent god knows how many $ in time to create this thing that will save me from buying 50$ worth of additional memory for my home machine.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from tinyvec.langchain_store import LangchainVectorDB

with open("dset.txt", "r") as f:
    articles = f.readlines()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=0,
    pipeline_kwargs={"max_new_tokens": 512},
)
embeddings = HuggingFaceEmbeddings(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
vectorstore = LangchainVectorDB.from_texts(
    articles, embedding=embeddings, emb_dim=2048, individually=True
)
retriever = vectorstore.as_retriever()

template = """
Answer the question in a full sentences giving full reasoning without
repetition based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("How long should I stretch if I want to grow my muscle size?"))
"""
> Answer: The recommended stretching routine for growing muscle size is 3 sets of 10-12 repetitions with 30 seconds rest in between. This routine can be done 3-4 times a week.
"""

print(chain.invoke("How should I optimize stretching for range of motion?"))
"""
Answer: The optimal stretching technique for range of motion depends on the individual's goals and the specific muscles being targeted. Here are some general guidelines:

1. Start with a warm-up: Before stretching, it's important to warm up your muscles with dynamic stretches. This will help prevent injury and improve flexibility.

2. Choose the right stretch: There are many different types of stretches, each with its own benefits and drawbacks. Choose a stretch that feels comfortable and safe for you.

3. Focus on the targeted muscles: Stretching the muscles that are causing pain or limiting movement can help improve range of motion.

4. Avoid overstretching: Overstretching can cause injury and reduce the benefits of stretching.

5. Gradually increase intensity: As you become more comfortable with a stretch, gradually increase the intensity.

6. Use props: If you're struggling to reach a certain stretch, consider using props like a foam roller, resistance band, or yoga block.

7. Monitor progress: Keep track of your progress and adjust your stretching routine as needed.

Remember, stretching is a tool to help you improve your range of motion, not a substitute for proper warm-up and injury prevention.
"""
```