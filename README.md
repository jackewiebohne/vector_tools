# vector_tools
to use for gensim model objects (esp. TWEC)

load the module (and any others that you might need)
```
from vector_tools import wordvec, vectorverse, metaverse
```
when a temporal TWEC model is trained, and say, you are interested in specific keywords and its nearest neighbors and how that's changed over time
you can create a metaverse of vectorverses, where each vectorverse contains the keyword (at a particular point in time) and its n nearest neighbors.
each keyword and its neighboring words in any vectorverse is represented as a wordvec object.

for example to read in trained models, look for keywords and their neighbors, convert to wordvecs and include in the vectorverses and metaverse, you can do this:

```
# keywords
keys = ['einbildung','einbilden','imaginieren','imagination','phantasie', 'phantasieren', 'urbild', 'fantasie', 'abbild']
# useful pandas dataframes with info about the texts/models
base_models = list(df3.textclass.unique())
base_models += list(df3.timebin.unique())
files = df.author[df.keyword_counts > 1].unique() # filter out those with keyword_count greater than x (arbitrary decision, though)
# grouped by timebin in ascending order: i.e. the closer to present time the higher the hierarchy (this is relevant for graphing the arrows in the metaverse)
hierarchy_dict = {row[1]:row[0] for row in grp[['timebin', "keyword_counts"]].itertuples()} 

vectorverses = []
errors = []
for file in pl.Path(wkdir + 'model/').iterdir():
    if file.is_file():
        name = nlp.get_filename(file,'.model') # a module used for 
        try:
            if name in base_models or name in files and not name.endswith('.npy'):
                model =  Word2Vec.load(str(file)) # Word2Vec is a gensim class!
                wvs = []
                for key in keys:
                    try:
                        tple = model.wv.most_similar(key, topn=10)
                        p = wordvec(key=name, word=key, value= model.wv[key])
                        for word, sim in tple:
                            wv = wordvec(key=name, word=word, value=model.wv[word], parent=p)
                            wvs.append(wv)
                        wvs.append(p)
                    except:
                        continue
    
                if wvs:
                    # create vectorverse: target_count == keyword_count; topn==10, 
                    # hierarchy only if model == one of the overall timebins (rather than an author or textclass based model)
                    tc = int(df3.keyword_counts[df3.author == name])
                    h = hierarchy_dict.get(name)
                    vv = vectorverse(key=name, wordvecs=wvs, hierarchy=h, topn=10, target_count=tc)
                    vectorverses.append(vv)
                
        except:
            print(name)
            errors.append(name)
```
the point of all this is predominantly for fast and easy dimensionality reduction and plotting

for example
```
timebinverse.reduce() # uses t-sne from scikit learn with default parameters (these can be changed by feeding params into the reduce method)
timebinverse.graph('phantasieren', 'over the centuries')
```
![example image](https://github.com/jackewiebohne/vector_tools/blob/main/example image.png?raw=true)
