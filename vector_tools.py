import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from adjustText import adjust_text
from matplotlib import pyplot as plt
import pathlib as pl
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def cos_similarity(x,y):
    return x@y/(np.linalg.norm(x)*np.linalg.norm(y))


class wordvec:
    '''
    takes a TWEC word, its vector, and its key:
    inputs
        - key: str: e.g. timebin or geolocation 
        - parent: str: the parent-word (as wordvec object) 
            under whose most similar words list the word appeared
        - word: str: the word, e.g. 'hello'
        - value: np.array(): the vector's value

    instance properties:
        self.key: timebin or geolocation 
        self.name: word, e.g. 'hello'
        self.val = the vector value
        self.dims = dimension of the vector
        self.score = similarity score to parent
    '''
    def __init__(self, key, word, value, parent=None, score=None):
        self.key = key # timebin or geolocation
        self.parent = parent # the word it was listed as among most similar under 
        self.name = word # the word 
        self.val = value # its vector value
        self.dims = value.shape[0] # dimension of the vector
        self.score = cos_similarity(value, parent.val) if parent and not score else score # similarity score to parent

    def __str__(self):
        if self.parent: 
            return f'word: {self.name}, first 3 values: {self.val[:3]}, total dims: {self.dims} \
            \nbelongs to key: {self.key} and has parent: {self.parent.name}'
        else:
            return f'word: {self.name}, first 3 values: {self.val[:3]}, total dims: {self.dims} \
            \nbelongs to key: {self.key} and is a parent'

    def setter(self, value):
        self.val = value
        self.dims = self.val.shape[0]

    def avg(self, other_vec, sep=None):
        '''
        to average this word vector with another one
        requires another wordvec object as input
        it works inplace
        further input:
            - sep: str or None (default: None): 
                separator to be placed between the wordvecs name
                and the other one with which it is averaged.
                e.g. 'hello_world' with sep=='_'
        '''
        if self.dims != other_vec.dims:
            raise ValueError('the vectors have unequal dimensions')
        if self.parent != other_vec.parent:
            raise ValueError('the vectors have unequal parents')
        if self.key != other_vec.key:
            print('WARNING: the vectors have different keys, but averaging anyway')
            if sep:
                self.key = self.key +sep+ other_vec.key
            else:
                self.key = self.name + other_vec.key
        if sep:
            self.name = self.name +sep+ other_vec.name
        else:
            self.name = self.name + other_vec.name
        self.val = (self.val + other_vec.val)/2
        self.score = cos_similarity(self.val, self.parent.val)

    def concat(self, other_vec, sep=None):
        '''
        to average this word vector with another one
        requires another wordvec object as input
        it works inplace
        further input:
            - sep: str or None (default: None): 
                separator to be placed between the wordvecs name
                and the other one with which it is averaged.
                e.g. 'hello_world' with sep=='_'
        '''
        if self.dims != other_vec.dims:
            raise ValueError('the vectors have unequal dimensions')
        if self.parent != other_vec.parent:
            raise ValueError('the vectors have unequal parents')
        if self.key != other_vec.key:
            print('WARNING: the vectors have different keys, but averaging anyway')
            if sep:
                self.key = self.key +sep+ other_vec.key
            else:
                self.key = self.name + other_vec.key
        if sep:
            self.name = self.name +sep+ other_vec.name
        else:
            self.name = self.name + other_vec.name
        self.val = np.concatenate((self.val, other_vec.val), axis=0)
        self.dims = self.val.shape[0]


class vectorverse:
    '''
    takes a gensim Word2Vec model
    and gives it graphing and other capabilities
    it may contain one or more parents (= keyword of interest) and their respective
    topn most similar words. but all of the parents and their most similar topn
    vectors belong to one location or timebin (depending on the TWEC training)
    
    inputs:
        key: str: the name of the vector verse
        wordvecs: list[wordvector]: list of wordvector objects
        topn: int: number of most similar vectors
        hierarchy: int (default: None): hierarchical position of vectorverse
            relative to other vectorverses in the metaverse. used for graphing
            because when graphing multiverse transitions between vectorverses will
            be connected with a line and an arrow.
        target_count: int or None (default: None):
            e.g. how often a particular word or group of words
            occurred in the vectorverse before training.
            this parameter can then later be used to select vectorverses
            according to a minimal count of a word or group of words of interest

    instance properties:
        self.key: the name of the vector verse 
            (if TWEC was used for training, this could e.g. be a timebin or geolocation)

        self.target_count: number of occurrence of a particular word or group of words
            occurred in the vectorverse.

    '''
    def __init__(self, key, wordvecs, hierarchy=None, topn=None, target_count=None):
        self.key = key
        self.parents = {p.name: p for p in set([wv.parent for wv in wordvecs if wv.parent])}
        self.wordvecs = [wv for wv in wordvecs if wv.parent] # can't use a dict because some words might appear multiple times
        self.size = len(self.wordvecs) + len(self.parents)
        self.topn = topn
        self.hierarchy = hierarchy 
        self.target_count = target_count
        self.dims = wordvecs[0].dims
        if not all(self.dims == wv.dims for wv in wordvecs):
            raise ValueError('Not all word vector dimensions are the same')
        self.reduced_with = None

    def __str__(self):
        return f'vectorverse {self.key} contains {self.size} elements with dimension {self.dims}.\
        \nit includes the parents {self.parents.keys()} and has topn number of similar words {self.topn} \
        \nit has a target count of {self.target_count} and has hierarchy position {self.hierarchy}'

    def __add__(self, wordvector):
        assert wordvector.dims == self.dims
        self.wordvecs.append(wordvector)
        if wordvector.parent.name not in self.parents.keys():
            self.parents[wordvector.parent.name] = wordvector.parent
        self.size = len(self.wordvecs) + len(self.parents)

    def save(self, out_path='./'):
        with open(out_path + f'vectorverse_{self.key}.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, in_path):
        raise NotImplementedError()
        # with open(in_path + '.pkl', 'rb') as inp:
        #     v = pickle.load(inp)

    def delete(self, wordvector):
        assert isinstance(wordvector, wordvec)
        p = wordvector.parent.name
        if p != None: # if wordvec is not a parent
            self.wordvecs = [wv for wv in self.wordvecs if not wv.name == wordvector.name and not wv.parent.name == wordvector.parent.name]
        else:
            del self.parents[p]
            self.wordvecs = [wv for wv in self.wordvecs if not wv.name == wordvector.name and not wv.parent.name == wordvector.parent.name]
        print(f'the wordvec {wordvector.name} was deleted successfully.\n\
            the target_count was not adjusted and will need to be done manually.')
        self.size = len(self.wordvecs) + len(self.parents)

    def get_parents(self, parent=[]):
        '''
        function to get parents in vectorverse:
        inputs:
            parent: list[str] (default: False): 
                if list the function will only return those parents matching
                the name of the parents, otherwise will return all parents
        '''
        if parent:
            return [v for k,v in self.parents.items() if k in parent]
        else:
            return list(self.parents.values())

    def get_children(self, parent=[]):
        '''
        function to get children of parents in vectorverse:
        inputs:
            parent: str or bool (default: False): 
                if str the function will only return those children with parents matching
                the name of the input parent(s), otherwise will return all children
        '''
        if not parent:
            return [wv for wv in self.wordvecs if wv.parent] # if wv.parent means it has a parent and can therefore not be itself a parent
        else:
            return [wv for wv in self.wordvecs if wv.parent and wv.parent.name in parent]


    def avg(self, wordvecs1, wordvecs2, sep=None):
        '''
        inputs:
            - wordvecs1: list[wordvec]: vectors to be averaged 
            - wordvecs2: list[wordvec]: vectors wordvecs1 are to be averaged with. 
                these will then be deleted from vectorverse
        '''
        assert(len(wordvec_names1) == len(wordvec_names2))
        self.wordvecs += list(self.parents.values())
        ix1 = [i for i, wv in enumerate(self.wordvecs) if (wv.name, wv.parent.name) in [(wv1.name, wv1.parent.name) for wv1 in wordvecs1]]
        ix2 = [j for j, wv in enumerate(self.wordvecs) if (wv.name, wv.parent.name) in [(wv2.name, wv2.parent.name) for wv2 in wordvecs2]]
        assert len(ix1) == len(ix2)
        any(self.wordvecs[ix1[i]].avg(self.wordvecs[ix2[i]], sep=sep) for i in range(len(ix1)))
        self.wordvecs = [wv for wv in self.wordvecs if wv.parent if (wv.name, wv.parent.name) not in [(wv2.name, wv2.parent.name) for wv2 in wordvecs2]]
        self.parents = {wv.parent.name: wv.parent for wv in self.wordvecs if wv.parent}
        self.size = len(self.wordvecs) + len(self.parents)

    def concat(): # should kick out 2nd word used in averaging from universe
        pass

    def reduce(self, n_components=2, **kwargs):
        '''
        uses t-SNE to reduce dimensionality by default
        note that an individual reduction of a vectorverse is NOT
        comparable to another vectorverse's individual reduction!
        for a combined and thus comparable dimensionality reduction add the vector-
        verses to one metaverse and reduce on the level of the metaverse!
        Inputs: 
            n_components: int: number of dimensions to keep
            kwargs: dict: dictionary containing string keys,
                        with variable values. possible key value pairs are:
                        {'method': 'tsne' or 'pca', 'perplexity': int}. if no
                        kwargs are provided it will perform tsne with defaults parameters
                        set by scikit learn. it is possible to provide your own method via kwargs.
                        in that case the reduce method will expect an already instantiated object under the key
                        'apply_func' in kwargs that can be called with 'fit_transform()'

        '''
        mtx = np.array([wv.val for wv in self.wordvecs])
        splice_idx = mtx.shape[0]
        ps = self.parents.values()
        mtxp = np.array([p.val for p in ps]).reshape(len(ps), -1)
        mtx = np.append(mtx, mtxp, axis=0)
        if kwargs and kwargs.get('method') == 'tsne':
            mtx = TSNE(n_components=n_components, perplexity=kwargs.get('perplexity')).fit_transform(mtx)
            self.reduced_with = 't-SNE'
        elif kwargs and kwargs.get('method') == 'pca':
            mtx = PCA(n_components=n_components).fit_transform(mtx)
            self.reduced_with = 'PCA'
        elif kwargs and kwargs.get('apply_func'):
            mtx = kwargs.get('apply_func').fit_transform(mtx)
            self.reduced_with = str(kwargs.get('apply_func'))
        else:
            mtx = TSNE(n_components=n_components).fit_transform(mtx)
            self.reduced_with = 't-SNE'
        any(wv.setter(mtx[i,:]) for i, wv in enumerate(self.wordvecs))
        mtxp = mtx[splice_idx:,:]
        ps = {p.name: mtxp[i,:] for i,p in enumerate(ps)}
        any(wv.parent.setter(ps.get(wv.parent.name)) for wv in self.wordvecs if wv.parent)
        self.dims = n_components

    def find_neighbors(self, potential_vectorverse_neighbors, parent_name, threshold, method):
        '''
        find neighboring vectorverses (that aren't part of this or another metaverse).
        what constitutes a neighbor is defined by the method.
        inputs:
            potential_vectorverse_neighbors: list[vectorverse]: list of vectorverses that could be neighbors
            threshold: int: how many neighbors should be found at most
            parent_name: str or list[str]: for which parent name(s) in the vectorverse should we find the neighbors
            method: str:
                if method == 'cos_similarity': it will look for the most cosine similar parents (up to at most threshold)
                among potential_vectorverse_neighbors
                if method == 'parent': it will use the input parent_name to find any parents in the input 
                potential_vectorverse_neighbors that share the parent_name as a substring. those will then be added as neighbors
        '''
        # add as property: self.neighbors = 0
        raise NotImplementedError()

    def graph(self, keyword, title=None, topn=5, p_color='red', c_color='blue', child_alpha=0.5, marker='.', save=False, show=True):
        '''
        graph function for vectorverse:
        inputs:
            keyword: str: which of parents to plot (displayed as red dot)
            title: str or None (default: None): title of figure
            topn: int (default: 5): number of n most similar vectors to parent to show
            p_color: str (default: 'red'): color of parent vectors
            c_color: str (default: 'blue'): color of children (=ordinary wordvecs)
            save: bool (default: False): if the figure should be saved
            show: bool or tuple(plt fig, plt axes) (default: True): 
                if just plotting this vectorverse True is the right option;
                if this vectorverse is plotted as part of the metaverse it will 
                internally get the fig and axes as input 
        ''' 
        if show==True:
            fig, axes = plt.subplots(figsize=(10, 10))
        else:
            fig, axes = show
        ### the most similar words vectors
        xs = []
        ys = []
        sims = self.wordvecs
        sims.sort(key=lambda wv: wv.score if wv.parent!=None and wv.score and wv.parent.name == keyword else 0, reverse=True)
        sims = [s for s in sims if s.score!=None][:topn]
        if sims:
            for wv in sims:
                assert wv.dims == 2
                xs.append(wv.val[0])
                ys.append(wv.val[1])
                axes.annotate(wv.name, (wv.val[0], wv.val[1]), c='dark'+c_color, alpha=child_alpha)
            axes.scatter(xs, ys, c=c_color, marker=marker, alpha=min(1, child_alpha * 1.2))
        ### the parent vectors
        if show == True: 
            if title:
                axes.set_title(title, fontsize=16, wrap=True)
            else:
                axes.set_title(f'{keyword} and its {topn} most similar words in {self.key}', fontsize=16, wrap=True)
            plt.xlabel(self.reduced_with+'_x')
            plt.ylabel(self.reduced_with + '_y')
            for wv in self.parents.values():
                if wv.name == keyword:
                    axes.plot(wv.val[0], wv.val[1], c=p_color, marker=marker)
                    axes.annotate(wv.name+'_'+wv.key, (wv.val[0], wv.val[1]),c='dark'+p_color, alpha=0.8)
        if save: fig.savefig('./' + keyword + f' {self.key}' + '_' + title+ '.jpg')
        if show==True: plt.show()



class metaverse:
    def __init__(self, superkey, vectorverses):
        if vectorverses:
            self.dims = vectorverses[0].dims
            self.topn = vectorverses[0].topn
            self.target_count = sum([vectorverse.target_count for vectorverse in vectorverses if vectorverse.target_count])
        else:
            self.dims = None
            self.topn = None
            self.target_count = 0
        if not all([self.topn == vectorverse.topn for vectorverse in vectorverses]) and vectorverses:
            print('WARNING: not all vectorverses have the same number topn!')
        self.superkey = superkey
        self.vectorverses = {vv.key: vv for vv in vectorverses}
        self.size = len(vectorverses)
        if not all([self.dims == v.dims for v in vectorverses]) and vectorverses:
            raise ValueError('Not all vectors are of same dimension across metaverse')
        self.reduced_with = None

    def __str__(self):
        string = f'the metaverse {self.superkey} contains {self.size} vectorverses:\n\n' 
        string += '\n\n'.join(f'vectorverse {i+1}\n{vectorverse}' for i, vectorverse in enumerate(self.vectorverses.values()))
        return string

    def __add__(self, vectorverse):
        if self.dims:
            assert self.dims == vectorverse.dims
        else:
            self.dims = vectorverse.dims
        if vectorverse.target_count:
            self.target_count += vectorverse.target_count
        if self.topn:
            if not vectorverse.topn == self.topn:
                print('WARNING: not all vectorverses have the same number topn!')
        else:
            self.topn = vectorverse.topn
        self.vectorverses[vectorverse.key] = vectorverse
        self.size = len(self.vectorverses.keys())

    def insert(self, vectorverse):
        '''
        function used to add vectorverse in a specific place in the metaverse
        with need to reshuffle the hierarchy
        '''
        pass

    def get_parents(self, parent=[]):
        if parent:
            return list(set([p for vv in self.vectorverses.values() for p in vv.parents.values() if p.name in parent]))
        else:
             return list(set([p for vv in self.vectorverses.values() for p in vv.parents.values()]))

    def get_children(self):
        # metaverse level function of eponymous vectorverse function
        raise NotImplementedError()

    def add_wordvec(self, wordvector, extent='all'):
        '''
        inputs:
            - wordvector: wordvec instance to be added
            - extent: st or list[str]: if 'all' adds the wordvec across all 
                vectorverses; otherwise input the key (as list of str) of the vectorverse(s)
                for addition of wordvec instance to just that vectorverse
        '''
        if extent == 'all':
            any(vv + wordvector for vv in self.vectorverses.values())
        else:
            any(vv + wordvector for k, vv in self.vectorverses.items() if k in extent) 

    def del_wordvec(self, wordvector, extent='all'):
        '''
        inputs:
            - wordvector: wordvec instance to be deleted
            - extent: str or list[str]: if 'all' deletes the wordvec across all 
                vectorverses; otherwise input the key (as list of str) of the vectorverse
                for deletion of wordvec instance in just that vectorverse
        '''
        if extent == 'all':
            any(vv.delete(wordvector) for vv in self.vectorverses.values())
        else:
            any(vv.delete(wordvector) for k, vv in self.vectorverses.items() if k in extent) 

    def implode(self, vecverse):
        '''
        going along (sort of) with the universe metaphor here,
        this function deletes a particular vectorverse
        '''
        if type(vecverse) == str:
            if self.vectorverses[vecverse].target_count and self.target_count:
                self.target_count -= self.vectorverses[vecverse].target_count
            del self.vectorverses[vecverse]
        elif isinstance(vecverse, vectorverse):
            if self.target_count and vecverse.target_count:
                self.target_count -= vecverse.target_count
            del self.vectorverses[vecverse.key]

    def avg(self, wordvec_names1, wordvec_names2, sep=None): # kicks out 2nd word used in averaging from universe
        '''
        inputs:
            - wordvec_names1: list[str]: vectors to be averaged 
            - wordvec_names2: list[str]: vectors wordvec_names1 are to be averaged with. 
                these will then be kicked out of all universes
        '''
        assert(len(wordvec_names1)==len(wordvec_names2))
        any(vectorverse.avg(wordvec_names1, wordvec_names2, sep=sep) for vectorverse in self.vectorverses.values())

    def concat():
        pass

    def reduce(self, n_components=2, **kwargs): 
        # TODO: give user choice to do operation inplace (replacing the values in the vectorverse 
        # by those reduced to n_components) or not and save the reduction as a copy

        '''
        uses t-SNE to reduce dimensionality by default
        this is the metaverse-level dimensionality reduction so that
        the reduced vectorverses it contains are comparable.
        Inputs: 
            n_components: int: number of dimensions to keep
            kwargs: dict: dictionary containing string keys,
                        with variable values. possible key value pairs are:
                        {'method': 'tsne' or 'pca', 'perplexity': int}. if no
                        kwargs are provided it will perform tsne with default parameters
                        set by scikit learn. it is possible to provide your own method via kwargs.
                        in that case the reduce method will expect an already instantiated object under the key
                        'apply_func' in kwargs that can be called with 'fit_transform()'

        '''
        mtx = 0
        splices = {i:[] for i in range(len(self.vectorverses.keys()))}
        for i, vv in enumerate(self.vectorverses.values()):
            if type(mtx) != int:
                splices[i].append(mtx.shape[0]) # start splice for wordvecs
                tmp = np.array([wv.val for wv in vv.wordvecs])
                mtx = np.append(mtx, tmp, axis=0)
            else:
                splices[i].append(0)
                mtx = np.array([wv.val for wv in vv.wordvecs])
            splices[i].append(mtx.shape[0]) # end splice for the wordvecs 
            ps = vv.parents.values()
            mtxp = np.array([p.val for p in ps]).reshape(len(ps), -1)
            mtx = np.append(mtx, mtxp, axis=0)
        # print(mtx)
        # mtx = TSNE(n_components=n_components, perplexity=30).fit_transform(mtx)
        if kwargs and kwargs.get('method') == 'tsne':
            mtx = TSNE(n_components=n_components, perplexity=kwargs.get('perplexity')).fit_transform(mtx)
            self.reduced_with = 't-SNE'
        elif kwargs and kwargs.get('method') == 'pca':
            mtx = PCA(n_components=n_components).fit_transform(mtx)
            self.reduced_with = 'PCA'
        elif kwargs and kwargs.get('apply_func'):
            mtx = kwargs.get('apply_func').fit_transform(mtx)
            self.reduced_with = str(kwargs.get('apply_func'))
        else:
            mtx = TSNE(n_components=n_components).fit_transform(mtx)
            self.reduced_with = 't-SNE'
        for i, vv in enumerate(self.vectorverses.values()):
            start, end = splices.get(i)
            any(wv.setter(mtx[start+j,:]) for j, wv in enumerate(vv.wordvecs))
            ps = {p.name: mtx[end+j,:] for j,p in enumerate(vv.parents.values())}
            any(wv.parent.setter(ps.get(wv.parent.name)) for wv in vv.wordvecs if wv.parent)
            vv.dims = n_components
        self.dims = n_components

    def _add_plot(self, to_plot, plt_objects,child_alpha, keyword, topn=None):
        '''
        internal function if any additional wordvecs are supposed to be added
        inputs:
            - to_plot: list[wordvec] or list[vectorverse]: 
                list of wordvecs or vectorverse to be added
            - plt_objects: tuple(plt fig, plt axes objects)
            - topn: int or None (default: None): 
                topn most simliar vectors to be added to plot for the parents
                if topn == 0 or None then only parents in vectorverse are added to plot
        '''
        fig, axes = plt_objects
        xs = []
        ys = []
        if isinstance(to_plot[0], wordvec):
            if topn:
                print('topn cannot be considered when adding simple wordvecs to plot.\
                 Add vectorverses instead')
            if not any(ele in [wv for vv in self.vectorverses.values() for wv in vv] for ele in to_plot):
                print('WARNING: the wordvecs to be plotted are not contained within the metaverse.\n\
                    if they were not two-dimensional before and have been reduced without alignment with the vectors of the metaverse their reduction is NOT representative')
            for wv in to_plot:
                assert wv.dims == 2
                xs.append(wv.val[0])
                ys.append(wv.val[1])
                axes.annotate(wv.name, (wv.val[0], wv.val[1]),c='green', alpha=child_alpha)
            axes.plot(xs, ys, c='green',marker='o')
        elif isinstance(to_plot[0], vectorverse):
            if not any(ele in self.vectorverses.values() for ele in to_plot):
                print('WARNING: the wordvecs of the vectorverse(s) to be plotted are not contained within the metaverse.\n\
                    if they were not two-dimensional before and have been reduced without alignment with the vectors of the metaverse their reduction is NOT representative')
            p_xs = []
            p_ys = []
            for vv in to_plot:
                ps = [vv.parents.get(name) for name in vv.parents.keys() if keyword in name][0]
                assert ps.dims == 2
                p_xs.append(ps.val[0])
                p_ys.append(ps.val[1])
                axes.annotate(ps.name+'_'+vv.key, (ps.val[0], ps.val[1]),c='brown', alpha=0.8)
                if topn:
                    vv.graph(keyword, topn=topn, c_color='green',marker='o', show=(fig, axes))
        if p_xs and p_ys:
            axes.plot(p_xs, p_ys, c='brown', marker='x')
        else:
            raise TypeError(f'elements in to_plot are of wrong type {type(to_plot[0])}\
                 but need to be either of type wordvec or vectorverse')

    def graph(self, keyword, title=None, topn=5, arrow_head_width=0.15, child_alpha=0.5,subset=False, save=False, add_plot=False):
        '''
        graph function for a single keyword in metaverse:
        inputs:
            keyword: str: which of parents to plot (displayed as red dot)
            title: str: title of figure
            topn: int (default: 5): number of n most similar vectors to parent to show
            arrow_head_width: float (default: 0.15): the width of the arrow-head that will be
                placed midway on the connecting line between two parent vectors in the
                neighboring vectorverses (neighbors = those whose hierarchy diff is 1). the size
                of the arrowhead might need significant adjusting dependent on how large the metaverse
                is (larger => smaller arrow-head)
            subset: bool or tuple(start:str, end:str) (default:False):
                if not False it plots only a subset of the vectorverses in the metaverse,
                sorted by the hierarchy of the vectorverses beginning with start and ending with end,
                where start and end are the keys (i.e. names) of the vectorverses
            save: bool (default: False): if the figure should be saved
            add_plot: bool or tuple(list[wordvecs] or list[vectorverse], topn:int) (default: False): 
                wordvecs or vectorverses to add to the plot that are not part of metaverse. 
                if the topn in tuple is larger than 0 and the list in tuple contains vectorverses, 
                the plot will also show topn number of most similar vectors to the parent vectors 
                in vectorverses. if topn == 0 then it will only
                plot the parent vectors in the respective vectorverses.
        ''' 
        fig, axes = plt.subplots(figsize=(20, 20))
        # sort vectorverses by hierarchy: the keyword-matching parent vectors in each vectorverse will 
        # then be connected by a line with arrow in the direction of ascending hierarchy
        hierarchical = [vv for vv in self.vectorverses.values() if vv.parents.get(keyword) and vv.hierarchy!=None]
        hierarchical.sort(key=lambda vv: vv.hierarchy if vv.hierarchy else 0)
        if subset:
            start, end = subset
            hierarchical = {vv.key: (i,vv) for i,vv in enumerate(hierarchical)}
            start = hierarchical.get(start)[0] # get the starting index for the subset
            end = hierarchical.get(end)[0]
            hierarchical = [vv for i, vv in hierarchical.values() if start<=i<=end]
        keymatrix = np.zeros((len(hierarchical), self.dims))
        for i, vv in enumerate(hierarchical):
            ps = vv.parents.get(keyword) # the assumption is that there's only one parent matching the keyword per vectorverse
            assert ps.dims == 2
            keymatrix[i,:] = ps.val 
            axes.annotate(vv.key, (ps.val[0], ps.val[1]),c='darkred', alpha=0.8) # if keyword and vectorverse name should be displayed replace vv.key with ps.name+'_'+vv.key
            vv.graph(keyword, child_alpha=child_alpha, topn=topn, show=(fig, axes))
        kx = keymatrix[:,0]
        ky = keymatrix[:,1]
        axes.plot(kx, ky, c='red', marker='.', alpha=0.8)
        if title:
            axes.set_title(title, fontsize=16, wrap=True)
        else:
            axes.set_title(f'{keyword} and its {topn} most similar words in {" and ".join([k for k in self.vectorverses.keys()])}', fontsize=16, wrap=True)
        ### the arrows on the keyvectors
        halfwayvecs = (keymatrix[1:,:] + keymatrix[:-1,:])/2 # gives the halfway point on the vector between one keyvector and the next
        i = 0
        while i < len(halfwayvecs):
            ### dennis's solution. many thanks!
            mid_x = halfwayvecs[i,0]
            mid_y = halfwayvecs[i,1]
            dx = (keymatrix[i+1,0] - keymatrix[i,0])/2 * 0.01
            dy = (keymatrix[i+1,1] - keymatrix[i,1])/2 * 0.01
            i+=1
            axes.arrow(x=mid_x, y=mid_y,
                     dx=dx, dy=dy,width=arrow_head_width, 
                     facecolor='red', edgecolor='none')
        plt.xlabel(self.reduced_with + '_x')
        plt.ylabel(self.reduced_with + '_y')
        if add_plot:
            lst, topm = add_plot
            self._add_plot(to_plot=lst, plt_objects=(fig, axes), keyword=keyword, child_alpha=child_alpha, topn=topm)
        if save: fig.savefig('./' + keyword + '_' + str(title)+ '.jpg')
        plt.show()

    def parallel_graph(matrix, indexdict, matrix2, indexdict2, keyword, title, arrow_head_width1=0.15,arrow_head_width2=0.15,save=False):
        raise NotImplementedError()
        keymatrix, keylist, matrix, indexdict = extract_arr_row(matrix, keyword, indexdict)
        keymatrix2, keylist2, matrix2, indexdict2 = extract_arr_row(matrix2, keyword, indexdict2)
        
        fig, ax = plt.subplots(figsize=(20, 20))
        
        xs = []
        ys = []
        ### points for one keyword timeline
        x1 = matrix[:,0]
        y1 = matrix[:,1]
        ax.scatter(x1, y1, c='r', alpha=0.5)
        plt.xlabel('t-SNE_x')
        plt.ylabel('t-SNE_y')
        texts1 = []
        for ix1, txt1 in enumerate(indexdict.values()):
            ax.annotate(txt1, (x1[ix1], y1[ix1]), c='r', alpha=0.8) # c='grey' alpha=0.7
            xs.append(x1[ix1])
            ys.append(y1[ix1])
    #         texts1.append(plt.text(x1[ix1], y1[ix1],txt1))

        ### line for one keyword timeline
        x2 = keymatrix[:,0]
        y2 = keymatrix[:,1]
        texts2 = []
        ax.plot(x2, y2, c='r')
        for ix2, txt2 in enumerate(keylist):
            texts2.append(plt.text(x2[ix2], y2[ix2],txt2, fontsize=12))
            xs.append(x2[ix2])
            ys.append(y2[ix2])
        
        ### points for the other keyword timeline
        x3 = matrix2[:,0]
        y3 = matrix2[:,1]
        texts3 = []
        ax.scatter(x3, y3, c='blue', alpha=0.5)
        for ix3, txt3 in enumerate(indexdict2.values()):
            ax.annotate(txt3, (x3[ix3], y3[ix3]), c='blue', alpha=0.8) # c='grey' alpha=0.7
            xs.append(x3[ix3])
            ys.append(y3[ix3])
    #         texts3.append(plt.text(x3[ix3], y3[ix3], txt3))
        
        ### line for the other keyword timeline
        x4 = keymatrix2[:,0]
        y4 = keymatrix2[:,1]
        texts4 = []
        ax.plot(x4, y4, c='blue')
        ax.set_title('\''+ keyword + '\' ' + title, fontsize=16, wrap=True)
        for ix4, txt4 in enumerate(keylist2):
            xs.append(x4[ix4])
            ys.append(y4[ix4])
            texts4.append(plt.text(x4[ix4], y4[ix4], txt4, fontsize=12))
        
        ### adding arrows for direction of first keyword timeline
        halfwayvecs1 = (keymatrix[1:,:] + keymatrix[:-1,:])/2 # gives the halfway point on the vector between one keyvector and the next
        i = 0
        while i < len(halfwayvecs1):
            ### dennis's solution. many thanks!
            mid_x = halfwayvecs1[i,0]
            mid_y = halfwayvecs1[i,1]
            dx = (keymatrix[i+1,0] - keymatrix[i,0])/2 * 0.01
            dy = (keymatrix[i+1,1] - keymatrix[i,1])/2 * 0.01
            i+=1
            ax.arrow(x=mid_x, y=mid_y,
                     dx=dx, dy=dy,width=arrow_head_width1, 
                     facecolor='red', edgecolor='none')
        
        ### adding arrows for direction of second keyword timeline
        halfwayvecs2 = (keymatrix2[1:,:] + keymatrix2[:-1,:])/2 # gives the halfway point on the vector between one keyvector and the next
        i = 0
        while i < len(halfwayvecs2):
            ### dennis's solution. many thanks!
            mid_x2 = halfwayvecs2[i,0]
            mid_y2 = halfwayvecs2[i,1]
            dx2 = (keymatrix2[i+1,0] - keymatrix2[i,0])/2 * 0.01
            dy2 = (keymatrix2[i+1,1] - keymatrix2[i,1])/2 * 0.01
            i+=1
            ax.arrow(x=mid_x2, y=mid_y2,
                     dx=dx2, dy=dy2,width=arrow_head_width2, 
                     facecolor='blue', edgecolor='none')
        
        
        ### avoiding text overlap
        adjust_text([*texts2, *texts4], x=np.array(xs), y=np.array(ys), precision=0.08,force_text=(0.5, 1) , arrowprops=dict(arrowstyle="->", color='black', lw=0.5)) # , only_move={'points':'y', 'texts':'y'}
    #     adjust_text(texts2, x=np.array(xs), y=np.array(ys), precision=0.08, force_text=(0.5, 1),arrowprops=dict(arrowstyle="->", color='r', lw=1)) # , only_move={'points':'y', 'texts':'y'}
    #     adjust_text(texts=[*texts2, *texts4],x=xs, y=ys , only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
        
        if save:
            fig.savefig('./parallel_' + keyword + '_' + str(title) + '.jpg')       


class bigbang:
    '''class for reading and saving models that will be turned into a metaverse'''
    def __init__(self):
        self.metaverse_name = None
        self.metaverse = None
        self.keywords = None
        self.topn = None
        self.hierarchy_dict = None

    def get_filename(self, file, return_filetype=False):
        f = str(file).split('\\')[-1].split('/')[-1][::-1]
        ix = f.find('.') # find last period
        name = f[ix+1:][::-1] 
        if return_filetype:
            ftype = f[:ix]
            return name, ftype
        else: return name

    def read_model(self, model_path, keywords, metaverse_name, topn, hierarchy_dict, exclude=[]):
        '''
        inputs:
            keywords: list[str]: keywords to look for in the model files
            model_path: str: folder path to the models
            topn: int: how many of the most similar vectors to grab
            hierarchy_dict: dict(str: int): a dictionary with the filename of the model
                    and its corresponding level in the hierarchy.
                    example: {'1900-1910': 1, '1910-1920': 2}
                    This hierarchy will then be used for plotting the metaverse
            exclude: any most similar words that should be excluded
        '''
        from gensim.models import Word2Vec
        self.keywords = keywords
        self.metaverse_name = metaverse_name
        self.topn = topn
        self.hierarchy_dict = hierarchy_dict
        vectorverses = []
        errors = []
        for file in pl.Path(model_path).iterdir():
            if file.is_file():
                name, ftype = self.get_filename(file, True)
                try:  
                    if name in hierarchy_dict.keys() and not ftype.endswith('npy') and not name.startswith('compass'):
                        model =  Word2Vec.load(str(file))
                        wvs = []
                        for key in self.keywords:
                                try:
                                    increment = 10 # 10 is arbitrary, it's chosen in case there's something to exclude
                                    tples = model.wv.most_similar(key, topn=self.topn + increment) 
                                    to_remove = sum([1 for t in tples if t[0] in exclude])
                                    if increment >= to_remove > 0:
                                        tples = [t for t in tples if t[0] not in exclude][:self.topn]
                                        assert len(tples) == self.topn
                                    else:
                                        while len(tples) < self.topn:
                                            tples = model.wv.most_similar(key, topn=self.topn + to_remove + increment)
                                            to_remove = sum([1 for t in tples if t[0] in exclude])
                                            increment += 10
                                            tples = [t for t in tples if t[0] not in exclude][:self.topn]
                                            assert len(tples) == self.topn
                                    p = wordvec(key=name, word=key, value= model.wv[key])
                                    wvs += [wordvec(key=name, word=word, value=model.wv[word], parent=p, score=sim) for word, sim in tples]
                                    wvs.append(p)
                                except: print(f'{key} not found in {name}')
                        if wvs:
                            h = hierarchy_dict.get(name)
                            vv = vectorverse(key=name, wordvecs=wvs, hierarchy=h, topn=self.topn)
                            vectorverses.append(vv)
                except: errors.append(name)
        if errors: print(f'encountered these errors: {errors}')
        self.metaverse = metaverse(superkey=self.metaverse_name, vectorverses=[])
        for vv in vectorverses:
            self.metaverse + vv
        return self.metaverse

    def create_keyed_similarity_df (self, keywords=[], save=False):
        ### TODO: give possibility to feed to bigbang a different metaverse than its own
        import pandas as pd
        if not self.metaverse: raise ValueError('please use read_model method first')
        sims = {}
        total_parents = [p.name for p in self.metaverse.get_parents(parent=keywords)]
        for k, v in self.metaverse.vectorverses.items():
            children = v.get_children(parent=keywords)
            dct = {p:[] for p in total_parents}
            for c in children:
                dct[c.parent.name] += [(c.name, c.score)]
            dct = {k: sorted(v, key=lambda x: x[1], reverse=True) for k,v in dct.items()}
            dct = {k: v[:self.topn] for k,v in dct.items()}
            assert all([len(v) == self.topn for v in dct.values()])
            sims[k] = dct
        sims = pd.DataFrame(sims)
        if not save:
            return sims
        else:
            sims.to_excel(save + 'keyed similarities.xlsx', engine='openpyxl') #sep='\t', index=False

    def save_metaverse(self, out_path='./'):
        import pickle
        ### TODO: give possibility to feed to bigbang a different metaverse than its own
        with open(out_path + self.metaverse_name + '.pkl', 'wb') as handle:
            pickle.dump(self.metaverse, handle, protocol=pickle.HIGHEST_PROTOCOL)


model_path = r"./joint_JSTOR_CORE\time_ordered_texts\original\model"
keywords = ['hustling', 'hustlers', 'hustler', 'hustle', 'hustles', 'hustled']
mname = 'hustling_through_time'
topn=10
hierarchy_dict = {str(i)+'-'+str(i+10): i for i in range(1900, 2020, 10)} # it doesn't actually matter what exact value hierarchy takes as long as there's an ordinal hierarchy
bb = bigbang()
bb.read_model(model_path=model_path, exclude =['tling', 'ofan', 'tliis', 'arly', 'pecially', 'afio', 'rial', 'auch', 'ithas'],
                             keywords=keywords, metaverse_name=mname, topn=topn, hierarchy_dict=hierarchy_dict)
# print(bb.create_keyed_similarity_df(save=r"C:\Users\jackewiebohne\Documents\python tests\DTA\joint_JSTOR_CORE/"))
mv = bb.metaverse
mv.reduce(2, **{'method': 'tsne', 'perplexity':33})
mv.graph('hustling', title='hustling from 1900 to 2020', child_alpha=0.5, arrow_head_width=0.15, topn=8)


# p = wordvec('1900-2000', 'sucks', np.array([1.8,1.8,1.8,1.8,1.8]))
# w = wordvec('1900-2000', 'helolo', np.array([2,2,2,2,2]), p)
# w2 = wordvec('1900-2000', 'helola', np.array([1,1,1,1,1]), p)
# verse = vectorverse('v',[p,w,w2],hierarchy=1)
# print(verse.dims)
# verse.reduce()
# print(verse.dims)
# print([wv.val for wv in verse.parents.values()])
# print([wv.val for wv in verse.wordvecs])
# p2 = wordvec('2000-2010', 'sucks', np.array([1.8,1.8,1.8,1.8,1.8]))
# w3 = wordvec('2000-2010', 'helolo', np.array([2,2,2,2,2]), p2)
# w4 = wordvec('2000-2010', 'helola', np.array([1,1,1,1,1]), p2)
# w5 = wordvec('2000-2010', 'heloli', np.array([1,1,1,1,1]), p2)
# verse2 = vectorverse('v2',[p2,w3,w4,w5], hierarchy=2)
# verse2 + wordvec('1900-2000', 'helo', np.array([1,1,2,1,1]), p2)
# verse2.avg(['helolo'], ['helola'],  sep='_averaged_with_')
# print(verse2.wordvecs)
# # print(verse2.wordvecs)
# # verse2.reduce()
# # print(verse2.size)
# # print(verse2.wordvecs['helolo'])
# # verse2.graph('fucker', 'the title')

# p3 = wordvec('1900-2000', 'glucks_sucks', np.array([1.8,1.9,1.7,1.5,3]))
# w6 = wordvec('1900-2000', 'ai', np.array([2,-2,1.7,1.5,3]), p3)
# verse3 = vectorverse('v3', [p3, w6])
# meta = metaverse('mega',[verse, verse2])
# meta #+ verse3
# print(meta)
# # meta.implode('v2')
# verse3.reduce()
# meta.reduce()
# print(meta)
# meta.graph('sucks','title', arrow_head_width=4, subset=('v2', 'v2'), add_plot=([verse3], 1))


# # meta.avg(['helolo'], ['helola'])
# # print(verse.wordvecs)
# # print(verse2.wordvecs)
# # print(verse,'\n', verse2)
# # print(verse.wordvecs['helolo'], '\n', verse2.wordvecs['helolo'])


# # print(timebinverse)
# timebinverse.reduce()
# timebinverse.graph('einbildung', 'title')
