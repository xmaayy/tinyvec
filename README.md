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