SUMMARY_STORY = """
<div class="story-box font-body">
<h2> About us</h2>
<p>
Hello everyone 👋, I am a <strong>News Article Summarizer</strong>. I'm here to give a little background of me and how we work here.
I was made by a group of <a href="https://huggingface.co/flax-community/t5-recipe-generation#team-members">UMBC Data Science Students</a> to train my two prodigy summarizers
who are well known in the Technology: <strong>T5 Transformer</strong> and <strong>BERT Transformer</strong>. 
The whole team participated in my rigorous training regiment of, <a href="https://huggingface.co/flax-community/t5-recipe-generation">T5 fine-tuning</a>, 
to learn how to get <strong>ABSTRACTIVE</strong> news summaries from any kind of News Articles. The students have tried really hard, invested a lot of time and 
almost gave up in the processing. But they were patient to pull themselves together and get this final product for you users.
I've never been more proud of my summarizers -- both can produce exceptional article summaries but I regard T5 as being <em>creative</em> while BERT is <em>meticulous</em>. 
If you give each of them the same articles, they'll usually come up with different summaries. <br /><br />
At the start of the program the summarizers read news articles containing thousands of news texts of different varieties like 'Extractive','Abstractive' and 'Mixed' formats
which were human written. 
The DataScience Students helped guide the learning process so that the summarizers could actually learn which articles work well together rather than just memorize and train
on everything. I trained my summarizers by asking them to generate a summary, for an article after giving them few sensible lines on the news piece. 
</p>
<pre>[Inputs]
    {News Article*: A complete CNN/Inshorts or any news article}
     
    [Targets]
    {News Summary*: A summarized format of the Above Article}
</pre>
<p>
  <em>The Summarizers were initially trained on <a href="https://lil.nlp.cornell.edu/newsroom/index.html"> Newsroom dataset</a>, the items the student trainers mostly
  concentrated were the Text, Summary and Format Variables for this training. </em>
</p>
<p>
In the span of a 4 weeks, my summarizers went from spitting out nonsense to creating masterpiece summaries. 
Their learning rate was exceptionally high and each batch of recipes was better than the last. <br />
Let me give you some more explaination on my Summarizers here on.
</p>
</div>
""".strip()
