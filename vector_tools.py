import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
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
    def __init__(self, key, word, value, parent=None):
        self.key = key # timebin or geolocation
        self.parent = parent # the word it was listed as among most similar under 
        self.name = word # the word 
        self.val = value # its vector value
        self.dims = value.shape[0] # dimension of the vector
        self.score = cos_similarity(value, parent.val) if parent else None # similarity score to parent

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

        self.target_count: 

    '''
    def __init__(self, key, wordvecs, hierarchy=None, topn=None, target_count=None):
        self.key = key
        self.parents = {p.name: p for p in set([wv.parent for wv in wordvecs if wv.parent])}
        self.wordvecs = {wv.name: wv for wv in wordvecs if wv.name not in self.parents.keys()}
        self.size = len(self.wordvecs) + len(self.parents)
        self.topn = topn
        self.hierarchy = hierarchy 
        self.target_count = target_count
        self.dims = wordvecs[0].dims
        if not all(self.dims == wv.dims for wv in wordvecs):
            raise ValueError('Not all word vector dimensions are the same')

    def __str__(self):
        return f'vectorverse {self.key} contains {self.size} elements with dimension {self.dims}.\
        \nit includes the parents {self.parents.keys()} and has topn number of similar words {self.topn} \
        \nit has a target count of {self.target_count} and has hierarchy position {self.hierarchy}'

    def __add__(self, wordvector):
        assert wordvector.dims == self.dims
        self.wordvecs[wordvector.name] = wordvector
        if wordvector.parent not in self.parents.keys():
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
            if type(wordvector) == str:
                p = self.wordvecs[wordvector].parent.name
                if any([parent.name==p for n, parent in self.parents.items()]):
                    del self.wordvecs[wordvector]
                else:
                    del self.parents[p]
                    del self.wordvecs[wordvector]
                print(f'the wordvec {wordvector} was deleted successfully.\n\
                    the target_count was not adjusted and will need to be done manually.')
            elif isinstance(wordvector, wordvec):
                p = wordvector.parent.name
                if any([parent.name==p for n, parent in self.parents.items()]):
                    del self.wordvecs[wordvector]
                else:
                    del self.parents[p]
                    del self.wordvecs[wordvector]
                print(f'the wordvec {wordvector.name} was deleted successfully.\n\
                    the target_count was not adjusted and will need to be done manually.')
            self.size = len(self.wordvecs) + len(self.parents)

    # def get_parents(self, parent=False):
    #     '''
    #     function to get parents in vectorverse:
    #     inputs:
    #         parent: str or bool (default: False): 
    #             if str the function will only return those parents matching
    #             the name of the parent, otherwise will return all parents
    #     '''
    #     if not parent:
    #         return set([wv.parent for wv in self.wordvecs.values() if wv.parent])
    #     else:
    #         return set([wv.parent for wv in self.wordvecs.values() if wv.parent and parent in wv.parent.name])

    def avg(self, wordvec_names1, wordvec_names2, sep=None):
        '''
        inputs:
            - wordvec_names1: list[str]: vectors to be averaged 
            - wordvec_names2: list[str]: vectors wordvec_names1 are to be averaged with. 
                these will then be deleted from vectorverse
        '''
        assert(len(wordvec_names1) == len(wordvec_names2))
        if set(wordvec_names1).intersection(wordvec_names2):
            raise ValueError(f'there is overlap between wordvec_names1 and wordvec_names2 which will lead to conflict\
                \noverlap: {set(wordvec_names1).intersection(wordvec_names2)}')
        self.wordvecs.update(self.parents)
        any(self.wordvecs[wordvec_names1[i]].avg(self.wordvecs[wordvec_names2[i]], sep=sep) \
            for i in range(len(wordvec_names1))\
            if wordvec_names1[i] in self.wordvecs.keys() and wordvec_names2[i] in self.wordvecs.keys())
        self.wordvecs = {k:v for k,v in self.wordvecs.items() if k not in wordvec_names2 and v.parent}
        self.parents = {wv.parent.name: wv.parent for wv in self.wordvecs.values() if wv.parent}
        self.size = len(self.wordvecs) + len(self.parents)

    def concat(): # should kick out 2nd word used in averaging from universe
        pass

    def reduce(self, n_components=2, perplexity=30.0):
        '''
        uses t-SNE to reduce dimensionality
        note that individual reduction of vectorverse doesn't
        means that it is comparable to another vectorverse's individual reduction
        for a combined and thus comparable dimensionality reduction add the vector-
        verses to one metaverse

        '''
        mtx = np.array([wv.val for wv in self.wordvecs.values()])
        splice_idx = mtx.shape[0]
        ps = self.parents.values()
        mtxp = np.array([p.val for p in ps]).reshape(len(ps), -1)
        mtx = np.append(mtx, mtxp, axis=0)
        mtx = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(mtx)
        any(wv.setter(mtx[i,:]) for i, wv in enumerate(self.wordvecs.values()))
        mtxp = mtx[splice_idx:,:]
        ps = {p.name: mtxp[i,:] for i,p in enumerate(ps)}
        any(wv.parent.setter(ps.get(wv.parent.name)) for wv in self.wordvecs.values() if wv.parent)
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

    def graph(self, keyword, title=None, topn=5, p_color='red', c_color='blue',marker='.', save=False, show=True):
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
        sims = list(self.wordvecs.values())
        sims.sort(key=lambda wv: wv.score if wv.parent!=None and wv.score and wv.parent.name == keyword else 0, reverse=True)
        sims = [s for s in sims if s.score!=None][:topn]
        if sims:
            for wv in sims:
                assert wv.dims == 2
                xs.append(wv.val[0])
                ys.append(wv.val[1])
                axes.annotate(wv.name, (wv.val[0], wv.val[1]), c=c_color, alpha=0.8)
            axes.scatter(xs, ys, c=c_color, marker=marker)
        ### the parent vectors
        if show == True: 
            axes.set_title('\''+ keyword + '\' ' + title, fontsize=16)
            plt.xlabel('t-SNE_x')
            plt.ylabel('t-SNE_y')
            for wv in self.parents.values():
                if wv.name == keyword:
                    axes.plot(wv.val[0], wv.val[1], c=p_color, marker=marker)
                    axes.annotate(wv.name+'_'+wv.key, (wv.val[0], wv.val[1]),c=p_color, alpha=0.8)
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

    def save(self, out_path='./'):
        pass

    def load(self, in_path):
        pass

    def insert(self, vectorverse):
        '''
        function used to add vectorverse in a specific place in the metaverse
        with need to reshuffle the hierarchy
        '''
        pass

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

    def avg(self, wordvec_names1, wordvec_names2, sep=None): # should kick out 2nd word used in averaging from universe
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

    def reduce(self, n_components=2, perplexity=30.0): 
        # use tSNE to reduce dimensionality
        mtx = 0
        splices = {i:[] for i in range(len(self.vectorverses.keys()))}
        for i, vv in enumerate(self.vectorverses.values()):
            if type(mtx) != int:
                splices[i].append(mtx.shape[0]) # start splice for wordvecs
                tmp = np.array([wv.val for wv in vv.wordvecs.values()])
                mtx = np.append(mtx, tmp, axis=0)
            else:
                splices[i].append(0)
                mtx = np.array([wv.val for wv in vv.wordvecs.values()])
            splices[i].append(mtx.shape[0]) # end splice for the wordvecs 
            ps = vv.parents.values()
            mtxp = np.array([p.val for p in ps]).reshape(len(ps), -1)
            mtx = np.append(mtx, mtxp, axis=0)
        mtx = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(mtx)
        for i, vv in enumerate(self.vectorverses.values()):
            start, end = splices.get(i)
            any(wv.setter(mtx[start+j,:]) for j, wv in enumerate(vv.wordvecs.values()))
            ps = {p.name: mtx[end+j,:] for j,p in enumerate(vv.parents.values())}
            any(wv.parent.setter(ps.get(wv.parent.name)) for wv in vv.wordvecs.values() if wv.parent)
            vv.dims = n_components
        self.dims = n_components

    def _add_plot(self, to_plot, plt_objects, keyword, topn=None):
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
                axes.annotate(wv.name, (wv.val[0], wv.val[1]),c='green', alpha=0.8)
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

    def graph(self, keyword, title, topn=5, arrow_head_width=0.15, subset=False, save=False, add_plot=False):
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
            axes.annotate(ps.name+'_'+vv.key, (ps.val[0], ps.val[1]),c='red', alpha=0.8)
            vv.graph(keyword, topn=topn, show=(fig, axes))
        kx = keymatrix[:,0]
        ky = keymatrix[:,1]
        axes.plot(kx, ky, c='red', marker='.')
        axes.set_title('\''+ keyword + '\' ' + title, fontsize=16)
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
        plt.xlabel('t-SNE_x')
        plt.ylabel('t-SNE_y')
        if add_plot:
            lst, topm = add_plot
            self._add_plot(to_plot=lst, plt_objects=(fig, axes), keyword=keyword, topn=topm)
        if save: fig.savefig('./' + keyword + '_' + title+ '.jpg')
        plt.show()

    def parallel_graph(matrix, indexdict, matrix2, indexdict2, keyword, title, arrow_head_width1=0.15,arrow_head_width2=0.15,save=False):
        raise NotImplementedError()
#         keymatrix, keylist, matrix, indexdict = extract_arr_row(matrix, keyword, indexdict)
#         keymatrix2, keylist2, matrix2, indexdict2 = extract_arr_row(matrix2, keyword, indexdict2)
        
#         fig, ax = plt.subplots(figsize=(20, 20))
        
#         xs = []
#         ys = []
#         ### points for one keyword timeline
#         x1 = matrix[:,0]
#         y1 = matrix[:,1]
#         ax.scatter(x1, y1, c='r', alpha=0.5)
#         plt.xlabel('t-SNE_x')
#         plt.ylabel('t-SNE_y')
#         texts1 = []
#         for ix1, txt1 in enumerate(indexdict.values()):
#             ax.annotate(txt1, (x1[ix1], y1[ix1]), c='r', alpha=0.8) # c='grey' alpha=0.7
#             xs.append(x1[ix1])
#             ys.append(y1[ix1])
#     #         texts1.append(plt.text(x1[ix1], y1[ix1],txt1))

#         ### line for one keyword timeline
#         x2 = keymatrix[:,0]
#         y2 = keymatrix[:,1]
#         texts2 = []
#         ax.plot(x2, y2, c='r')
#         for ix2, txt2 in enumerate(keylist):
#             texts2.append(plt.text(x2[ix2], y2[ix2],txt2, fontsize=12))
#             xs.append(x2[ix2])
#             ys.append(y2[ix2])
        
#         ### points for the other keyword timeline
#         x3 = matrix2[:,0]
#         y3 = matrix2[:,1]
#         texts3 = []
#         ax.scatter(x3, y3, c='blue', alpha=0.5)
#         for ix3, txt3 in enumerate(indexdict2.values()):
#             ax.annotate(txt3, (x3[ix3], y3[ix3]), c='blue', alpha=0.8) # c='grey' alpha=0.7
#             xs.append(x3[ix3])
#             ys.append(y3[ix3])
#     #         texts3.append(plt.text(x3[ix3], y3[ix3], txt3))
        
#         ### line for the other keyword timeline
#         x4 = keymatrix2[:,0]
#         y4 = keymatrix2[:,1]
#         texts4 = []
#         ax.plot(x4, y4, c='blue')
#         ax.set_title('\''+ keyword + '\' ' + title, fontsize=16)
#         for ix4, txt4 in enumerate(keylist2):
#             xs.append(x4[ix4])
#             ys.append(y4[ix4])
#             texts4.append(plt.text(x4[ix4], y4[ix4], txt4, fontsize=12))
        
#         ### adding arrows for direction of first keyword timeline
#         halfwayvecs1 = (keymatrix[1:,:] + keymatrix[:-1,:])/2 # gives the halfway point on the vector between one keyvector and the next
#         i = 0
#         while i < len(halfwayvecs1):
#             ### dennis's solution. many thanks!
#             mid_x = halfwayvecs1[i,0]
#             mid_y = halfwayvecs1[i,1]
#             dx = (keymatrix[i+1,0] - keymatrix[i,0])/2 * 0.01
#             dy = (keymatrix[i+1,1] - keymatrix[i,1])/2 * 0.01
#             i+=1
#             ax.arrow(x=mid_x, y=mid_y,
#                      dx=dx, dy=dy,width=arrow_head_width1, 
#                      facecolor='red', edgecolor='none')
        
#         ### adding arrows for direction of second keyword timeline
#         halfwayvecs2 = (keymatrix2[1:,:] + keymatrix2[:-1,:])/2 # gives the halfway point on the vector between one keyvector and the next
#         i = 0
#         while i < len(halfwayvecs2):
#             ### dennis's solution. many thanks!
#             mid_x2 = halfwayvecs2[i,0]
#             mid_y2 = halfwayvecs2[i,1]
#             dx2 = (keymatrix2[i+1,0] - keymatrix2[i,0])/2 * 0.01
#             dy2 = (keymatrix2[i+1,1] - keymatrix2[i,1])/2 * 0.01
#             i+=1
#             ax.arrow(x=mid_x2, y=mid_y2,
#                      dx=dx2, dy=dy2,width=arrow_head_width2, 
#                      facecolor='blue', edgecolor='none')
        
        
#         ### avoiding text overlap
#         adjust_text([*texts2, *texts4], x=np.array(xs), y=np.array(ys), precision=0.08,force_text=(0.5, 1) , arrowprops=dict(arrowstyle="->", color='black', lw=0.5)) # , only_move={'points':'y', 'texts':'y'}
    #     adjust_text(texts2, x=np.array(xs), y=np.array(ys), precision=0.08, force_text=(0.5, 1),arrowprops=dict(arrowstyle="->", color='r', lw=1)) # , only_move={'points':'y', 'texts':'y'}
    #     adjust_text(texts=[*texts2, *texts4],x=xs, y=ys , only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
        
        if save:
            fig.savefig('./parallel_' + keyword + '_' + title + '.jpg')       
