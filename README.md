# Phys-489-Project-temp-title

This is a research project on new metrics for evaluating research papers. We limited our scope to physics papers only.

Steps:

~ Pull as much data as possible from openAlex. I used an OpenAlex scraper for this. We are also writing scripts to do that.


~ Next, we used the DOIs to check if the paper has been retracted using CrossRef or Retraction Watch.


~ Check if data is available (i.e RDM)


~ Start comparing abstracts using EmbeddingGemma


~ Pull the H-index of authors and compare it to the calculated H-index based on their works available through OpenAlex


~ Compare the Abstract vector embeddings of all of a given author's works with their H-index
