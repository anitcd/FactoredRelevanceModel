/*
TODO: To add an option:
    If Rerank (i.e. KLD), then add an option whether to precompute the 
        collection statistics (e.g. col-probab. that is needed) for all the terms
        of collection into hashmap to reduce time consumption during reranking.
 */
package RelevanceFeedback;

import static common.CommonVariables.FIELD_ID;
import common.DocumentVector;
import common.PerTermStat;
import common.TRECQuery;
import common.TRECQueryParser;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.lang.ArrayUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import searcher.BERT;
import searcher.BERTVec;
import searcher.ContextualQuery;
import searcher.TRECCSQuery;
import searcher.TermList;
import searcher.UserPreference;
import searcher.WebDocSearcher_TRECCS_Novel;
import searcher.Word2vec;

/**
 *
 * @author base RM3 implementation by dwaipayan. Extented to (KDE)FRLM by anirban.
 */
public class RLM {

    IndexReader     indexReader;
    IndexSearcher   indexSearcher;
    String          fieldForFeedback;   // the field of the index which will be used for feedback
    Analyzer        analyzer;
    
    public int      numFeedbackTerms;// number of feedback terms
    public int      numFeedbackDocs; // number of feedback documents
    float           mixingLambda;    // mixing weight, used for doc-col weight adjustment
    float           QMIX;           // query mixing parameter; to be used for RM3, RM4 (not done)

    RelevanceBasedLanguageModel rblm;   // main class from which the call is done; used for setting the variables.

    /**
     * Hashmap of Vectors of all feedback documents, keyed by luceneDocId.
     */
    HashMap<Integer, DocumentVector>    feedbackDocumentVectors;
    /**
     * HashMap of PerTermStat of all feedback terms, keyed by the term.
     */
    HashMap<String, PerTermStat>        feedbackTermStats;
    /**
     * HashMap of P(Q|D) for all feedback documents, keyed by luceneDocId.
     */
    HashMap<Integer, Float> hash_P_Q_Given_D;

    TopDocs         topDocs;

    long            vocSize;        // vocabulary size
    long            docCount;       // number of documents in the collection
    
    TRECQueryParser         trecQueryparser;
    List<String []>         W2Vmodel;
    List<Word2vec>          W2V;
    List<BERT>              bert;
    List<TRECCSQuery>       treccsQueryJson;
    List<ContextualQuery>   contextualApproTerms;

    /**
     * List, for sorting the words in non-increasing order of probability.
     */
    List<WordProbability> list_PwGivenR;
    /**
     * HashMap of P(w|R) for 'numFeedbackTerms' terms with top P(w|R) among each w in R,
     * keyed by the term with P(w|R) as the value.
     */
    HashMap<String, WordProbability> hashmap_PwGivenR;

    /**
     * HashMap<DocId, DocumentVector> to contain all topdocs for reranking.
     * Only used if reranking, reading top docs from file.
     */
    HashMap<String, DocumentVector> topDocsDV = new HashMap<>();
    
    // Ani... to use from WebDocSearcher_TRECCS_Novel.java
    public RLM(IndexReader indexReader, IndexSearcher indexSearcher, Analyzer analyzer, String fieldForFeedback, int numFeedbackDocs, int numFeedbackTerms, float QMIX, float param1, List<String []> W2Vmodel, List<Word2vec> W2V, List<BERT> bert, TRECQueryParser trecQueryparser, List<TRECCSQuery> treccsQueryJson, List<ContextualQuery> contextualApproTerms) throws IOException {
        this.indexReader = indexReader;
        this.indexSearcher = indexSearcher;
        this.analyzer = analyzer;
        this.fieldForFeedback = fieldForFeedback;
        this.numFeedbackDocs = numFeedbackDocs;
        this.numFeedbackTerms = numFeedbackTerms;
        this.QMIX = QMIX;
        if(param1>0.99)
            this.mixingLambda = 0.8f;
        else
            this.mixingLambda = param1;

        vocSize = getVocabularySize();
        docCount = indexReader.maxDoc();      // total number of documents in the index

        this.W2Vmodel = W2Vmodel;
        this.W2V = W2V;
        this.bert = bert;
        this.trecQueryparser = trecQueryparser;
        this.treccsQueryJson = treccsQueryJson;
        this.contextualApproTerms = contextualApproTerms;
    }

    public RLM(RelevanceBasedLanguageModel rblm) throws IOException {

        this.rblm = rblm;
        this.indexReader = rblm.indexReader;
        this.indexSearcher = rblm.indexSearcher;
        this.analyzer = rblm.analyzer;
        this.fieldForFeedback = rblm.fieldForFeedback;
        this.numFeedbackDocs = rblm.numFeedbackDocs;
        this.numFeedbackTerms = rblm.numFeedbackTerms;
        this.mixingLambda = rblm.mixingLambda;
        this.QMIX = rblm.QMIX;
        vocSize = getVocabularySize();
        docCount = indexReader.maxDoc();      // total number of documents in the index

    }

    // Ani... to use from WebDocSearcher_TRECCS_Novel.java
    /**
     * Sets the following variables with feedback statistics: to be used consequently.<p>
     * {@link #feedbackDocumentVectors},<p> 
     * {@link #feedbackTermStats}, <p>
     * {@link #hash_P_Q_Given_D}
     * @param topDocs
     * @param analyzedQuery
     * @param H_R_Choice: 0 when using user History 'H+' as relevance model. Note: We want to used all docs from history
     *                    1 when using top retrieved docs as relevance model. Note: We want to use top 'numFeedbackDocs' docs
     * @throws IOException 
     */
    public void setFeedbackStatsDirect(TopDocs topDocs, String[] analyzedQuery, int H_R_Choice) throws Exception {
        
        feedbackDocumentVectors = new HashMap<>();
        feedbackTermStats = new HashMap<>();
        hash_P_Q_Given_D = new HashMap<>();

        ScoreDoc[] hits;
        int hits_length;
        hits = topDocs.scoreDocs; // Doi
        //hits = normalizeMinMax_hits(topDocs.scoreDocs); // Ani
        hits_length = hits.length;               // number of documents retrieved in the first retrieval
        
        int nDocs;
        if(H_R_Choice == 0)
            nDocs = hits_length;
        else
            nDocs = Math.min(numFeedbackDocs, hits_length);

        for (int i = 0; i < nDocs; i++) {
            // for each feedback document
            int luceneDocId = hits[i].doc;
            Document d = indexSearcher.doc(luceneDocId);
            DocumentVector docV = new DocumentVector(this.fieldForFeedback);
            //DocumentVector docV = new DocumentVector(this.fieldForFeedback, hits[i].score); // Ani... adding retrieval score of the doc
            //docV = docV.getDocumentVector(luceneDocId, indexReader);
            docV = docV.getDocumentVector(luceneDocId, indexReader, hits[i].score);  // Ani... adding retrieval score of the doc
            if(docV == null)
                continue;
            feedbackDocumentVectors.put(luceneDocId, docV);                // the document vector is added in the list

            for (Map.Entry<String, PerTermStat> entrySet : docV.docPerTermStat.entrySet()) {
            // for each term of that feedback document
                String key = entrySet.getKey();
                PerTermStat value = entrySet.getValue();

                // ++ This is giving better MAP for TREC 9, compared to the next commented snippet, BUT NOT WITH OTHER TOPIC 
//                if(null == feedbackTermStats.get(key))
//                    feedbackTermStats.put(key, new PerTermStat(key, value.getCF(), value.getDF()));
//                else {
//                    value.setCF(value.getCF() + feedbackTermStats.get(key).getCF());
//                    value.setDF(value.getDF() + feedbackTermStats.get(key).getDF());
//                    feedbackTermStats.put(key, value);
//                }
                // -- commented the following
                if(null == feedbackTermStats.get(key)) {
                // this feedback term is not already put in the hashmap, hence needed to be put;
                    Term termInstance = new Term(fieldForFeedback, key);
                    long cf = indexReader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).
                    long df = indexReader.docFreq(termInstance);       // DF: Returns the number of documents containing the term

//                    feedbackTermStats.put(key, new PerTermStat(key, value.getCF(), value.getDF()));
                    feedbackTermStats.put(key, new PerTermStat(key, cf, df));
                }
            } // ends for each term of that feedback document
        } // ends for each feedback document

        // Calculating P(Q|d) for each feedback documents
        for (Map.Entry<Integer, DocumentVector> entrySet : feedbackDocumentVectors.entrySet()) {
            // for each feedback document
            int luceneDocId = entrySet.getKey();
            DocumentVector docV = entrySet.getValue();

            float p_Q_GivenD = 1;
            for (String qTerm : analyzedQuery)
                p_Q_GivenD *= return_Smoothed_MLE(qTerm, docV);
            if(null == hash_P_Q_Given_D.get(luceneDocId))
                hash_P_Q_Given_D.put(luceneDocId, p_Q_GivenD);
            else {
                System.err.println("Error while pre-calculating P(Q|d). "
                + "For luceneDocId: " + luceneDocId + ", P(Q|d) already existed.");
            }
        }

    }

    /**
     * Sets the following variables with feedback statistics: to be used consequently.<p>
     * {@link #feedbackDocumentVectors},<p> 
     * {@link #feedbackTermStats}, <p>
     * {@link #hash_P_Q_Given_D}
     * @param topDocs
     * @param analyzedQuery
     * @param rblm
     * @throws IOException 
     */
    public void setFeedbackStats(TopDocs topDocs, String[] analyzedQuery, RelevanceBasedLanguageModel rblm) throws IOException {

        feedbackDocumentVectors = new HashMap<>();
        feedbackTermStats = new HashMap<>();
        hash_P_Q_Given_D = new HashMap<>();

        ScoreDoc[] hits;
        int hits_length;
        hits = topDocs.scoreDocs;
        hits_length = hits.length;               // number of documents retrieved in the first retrieval

        for (int i = 0; i < Math.min(numFeedbackDocs, hits_length); i++) {
            // for each feedback document
            int luceneDocId = hits[i].doc;
            Document d = indexSearcher.doc(luceneDocId);
            DocumentVector docV = new DocumentVector(rblm.fieldForFeedback);
            docV = docV.getDocumentVector(luceneDocId, indexReader);
            if(docV == null)
                continue;
            feedbackDocumentVectors.put(luceneDocId, docV);                // the document vector is added in the list

            for (Map.Entry<String, PerTermStat> entrySet : docV.docPerTermStat.entrySet()) {
            // for each term of that feedback document
                String key = entrySet.getKey();
                PerTermStat value = entrySet.getValue();

                // ++ This is giving better MAP for TREC 9, compared to the next commented snippet, BUT NOT WITH OTHER TOPIC 
//                if(null == feedbackTermStats.get(key))
//                    feedbackTermStats.put(key, new PerTermStat(key, value.getCF(), value.getDF()));
//                else {
//                    value.setCF(value.getCF() + feedbackTermStats.get(key).getCF());
//                    value.setDF(value.getDF() + feedbackTermStats.get(key).getDF());
//                    feedbackTermStats.put(key, value);
//                }
                // -- commented the following
                if(null == feedbackTermStats.get(key)) {
                // this feedback term is not already put in the hashmap, hence needed to be put;
                    Term termInstance = new Term(fieldForFeedback, key);
                    long cf = indexReader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).
                    long df = indexReader.docFreq(termInstance);       // DF: Returns the number of documents containing the term

//                    feedbackTermStats.put(key, new PerTermStat(key, value.getCF(), value.getDF()));
                    feedbackTermStats.put(key, new PerTermStat(key, cf, df));
                }
            } // ends for each term of that feedback document
        } // ends for each feedback document

        // Calculating P(Q|d) for each feedback documents
        for (Map.Entry<Integer, DocumentVector> entrySet : feedbackDocumentVectors.entrySet()) {
            // for each feedback document
            int luceneDocId = entrySet.getKey();
            DocumentVector docV = entrySet.getValue();

            float p_Q_GivenD = 1;
            for (String qTerm : analyzedQuery)
                p_Q_GivenD *= return_Smoothed_MLE(qTerm, docV);
            if(null == hash_P_Q_Given_D.get(luceneDocId))
                hash_P_Q_Given_D.put(luceneDocId, p_Q_GivenD);
            else {
                System.err.println("Error while pre-calculating P(Q|d). "
                + "For luceneDocId: " + luceneDocId + ", P(Q|d) already existed.");
            }
        }

    }

    /**
     * mixingLambda*tf(t,d)/d-size + (1-mixingLambda)*cf(t)/col-size
     * log(1+mixingLambda/(1-mixingLambda)*tf(t,d)/d-size*col-size/cf(t)) // DG try
     * @param t The term under consideration
     * @param dv The document vector under consideration
     * @return MLE of t in a document dv, smoothed with collection statistics
     */
    public float return_Smoothed_MLE(String t, DocumentVector dv) throws IOException {

        float smoothedMLEofTerm = 1;
        PerTermStat docPTS;

//        HashMap<String, PerTermStat>     docPerTermStat = dv.getDocPerTermStat();
//        docPTS = docPerTermStat.get(t);
        docPTS = dv.docPerTermStat.get(t);
//        colPTS = collStat.perTermStat.get(t);
        PerTermStat colPTS = feedbackTermStats.get(t);

        if (colPTS != null) {
            smoothedMLEofTerm = 
                //((docPTS!=null)?((float)docPTS.getCF()):(0));
                //((docPTS!=null)?((float)docPTS.getCF() / (float)dv.getDocSize()):(0)); // Ani - without smoothing. For smoothing open next two lines
                ((docPTS!=null)?(mixingLambda * (float)docPTS.getCF() / (float)dv.getDocSize()):(0)) +
                ((feedbackTermStats.get(t)!=null)?((1.0f-mixingLambda)*(float)feedbackTermStats.get(t).getCF()/(float)vocSize):0); // Doi
                //(float) Math.log(1+mixingLambda/(1-mixingLambda) * (float)docPTS.getCF() / (float)dv.getDocSize() * (float)vocSize / (float)feedbackTermStats.get(t).getCF()); // DG

//            System.out.print(t + "\tsmoothedMLEofTerm: " + smoothedMLEofTerm);
//            smoothedMLEofTerm = (float) Math.log(1+mixingLambda/(1-mixingLambda) * (float)docPTS.getCF() / (float)dv.getDocSize() * (float)vocSize / (float)feedbackTermStats.get(t).getCF()); // DG
//            //System.out.println("smoothedMLEofTerm: " + smoothedMLEofTerm);
//            System.out.print("\t" + smoothedMLEofTerm + "\t");
//            System.out.println("TF: " + docPTS.getCF() + ", docLen: " + dv.getDocSize() + ", CF: " + feedbackTermStats.get(t).getCF() + ", CS: " + vocSize);
            //System.exit(1);
//            (1.0f-mixingLambda)*(getCollectionProbability(t, indexReader, fieldForFeedback));
        }
        return smoothedMLEofTerm;
    } // ends return_Smoothed_MLE()
    
    public float return_Smoothed_MLE_Log(String t, DocumentVector dv) throws IOException {
        
        float smoothedMLEofTerm = 1;
        PerTermStat docPTS;
        docPTS = dv.docPerTermStat.get(t);
        PerTermStat colPTS = feedbackTermStats.get(t);

        if (colPTS != null) 
            smoothedMLEofTerm = 
                ((docPTS!=null)?(mixingLambda * (float)docPTS.getCF() / (float)dv.getDocSize()):(0)) /
                ((feedbackTermStats.get(t)!=null)?((1.0f-mixingLambda)*(float)feedbackTermStats.get(t).getCF()/(float)vocSize):0);
     
        return (float)Math.log(1+smoothedMLEofTerm);

    } // ends return_Smoothed_MLE_Log()

    /**
     * Returns the vocabulary size of the collection for 'fieldForFeedback'.
     * @return vocSize Total number of terms in the vocabulary
     * @throws IOException IOException
     */
    private long getVocabularySize() throws IOException {

        Fields fields = MultiFields.getFields(indexReader);
        Terms terms = fields.terms(fieldForFeedback);
        if(null == terms) {
            System.err.println("Field: "+fieldForFeedback);
            System.err.println("Error buildCollectionStat(): terms Null found");
        }
        vocSize = terms.getSumTotalTermFreq();  // total number of terms in the index in that field

        return vocSize;  // total number of terms in the index in that field
    }
    
    public float getTF(String term, int luceneDocId) throws Exception {
        DocumentVector dv = feedbackDocumentVectors.get(luceneDocId);
        return dv.getTf(term, dv);
    }
    
    public int getDocLen(int luceneDocId) throws Exception {
        DocumentVector dv = feedbackDocumentVectors.get(luceneDocId);
        return dv.getDocSize();
    }
    
    // Geet P(w|M) where M (\mathcal_M) is feedback docs
    public float get_P_wGivenM(String term) throws Exception {
        float P_wGivenM = 0.0f, P_wGivenM1 = 0.0f, docSize = 0.0f;
        //int nDoc = feedbackDocumentVectors.size();
        for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
            int luceneDocId = docEntrySet.getKey();
            //DocumentVector docV = docEntrySet.getValue();

            //P_wGivenM += getTF(term, luceneDocId);
            //P_wGivenM += (getTF(term, luceneDocId) / (float) getDocLen(luceneDocId));
            //docSize += (float) getDocLen(luceneDocId);
            //P_wGivenM += return_Smoothed_MLE(term, feedbackDocumentVectors.get(luceneDocId));
            P_wGivenM += return_Smoothed_MLE_Log(term, feedbackDocumentVectors.get(luceneDocId));
            //System.out.println(term + "\tTF: " + P_wGivenM + "\tlen: " + docSize);
        }
        //P_wGivenM /= docSize;
//        System.out.println("P_wGivenM: " + P_wGivenM);
//        System.out.println("P_wGivenM1: " + P_wGivenM1);
//        System.exit(1);
        return P_wGivenM;
        //return P_wGivenM1;
    }
    
    public float getCollectionProbability(String term, IndexReader reader, String fieldName) throws IOException {

        Term termInstance = new Term(fieldName, term);
        long termFreq = reader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).

        return (float) termFreq / (float) vocSize;
    }

    public float reciprocalCollProb(String term, IndexReader reader, String fieldName) throws IOException {
	return 1/getCollectionProbability(term, reader, fieldName);
    }

    /**
     * Returns MLE of a query term q in Q;<p>
     * P(w|Q) = tf(w,Q)/|Q|
     * @param qTerms all query terms
     * @param qTerm query term under consideration
     * @return MLE of qTerm in the query qTerms
     */
    public float returnMLE_of_q_in_Q(String[] qTerms, String qTerm) {

        int count=0;
        for (String queryTerm : qTerms)
            if (qTerm.equals(queryTerm))
                count++;
        return ( (float)count / (float)qTerms.length );
    } // ends returnMLE_of_w_in_Q()

    /**
     * RM1: IID Sampling <p>
     * Returns 'hashmap_PwGivenR' containing all terms of PR docs (PRD) with 
     * weights calculated using IID Sampling <p>
     * P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d)}
     * Reference: Relevance Based Language Model - Victor Lavrenko (SIGIR-2001)
     * @param query The query
     * @param topDocs Initial retrieved document list
     * @return 'hashmap_PwGivenR' containing all terms of PR docs with weights
     * @throws Exception 
     */
    ///*
    public HashMap RM1(TRECQuery query, TopDocs topDocs) throws Exception {

        float p_W_GivenR_one_doc;

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;
            float KDEScore = 1.0f;
            //float KDEScore = getKDEScore(t, query, 1.0f);
            //float KDEScore = getKDEScoreBERT(t, query, 1.0f);

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                float docScore = docEntrySet.getValue().getDocScore(); // Ani
                
                //System.out.println("(" + t + "): " + KDEScore);
                // Ani... ekhane KDERLM dhokate hobe. semantic similarity goon hobe
                p_W_GivenR_one_doc += 
                    //docScore * // mostly for 2D KDE (experimental)
                    return_Smoothed_MLE_Log(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId) *
                        ((KDEScore!=0.0f)?KDEScore:1); // Multiplying KDE socre of term 't' based on 'query' terms
            }
            
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }   // ends RM1()


    /**
     * RM3 <p>
     * P(w|R) = QueryMix*RM1 + (1-QueryMix)*P(w|Q) <p>
     * Reference: Nasreen Abdul Jaleel - TREC 2004 UMass Report <p>
     * @param query The query 
     * @param topDocs Initially retrieved document list
     * @return hashmap_PwGivenR: containing numFeedbackTerms expansion terms with normalized weights
     * @throws Exception 
     */
    public HashMap RM3(TRECQuery query, TopDocs topDocs) throws Exception {

        hashmap_PwGivenR = new LinkedHashMap<>();

        //hashmap_PwGivenR = RM1(query, topDocs);
        //hashmap_PwGivenR = KDERM1(query, topDocs);
        //hashmap_PwGivenR = KDERM1General(query, topDocs);
        //hashmap_PwGivenR = KDERM1_Doi(query, topDocs);
        hashmap_PwGivenR = KDERM1_2D_Doi(query, topDocs);
        // hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        
//        // +++ Inserting the idf factor
//        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
//            String key = entrySet.getKey();
//            WordProbability value = entrySet.getValue();
//            value.p_w_given_R *= Math.log(docCount/feedbackTermStats.get(key).getDF()+1);
//            hashmap_PwGivenR.put(key, value);
//        }
//        hashmap_PwGivenR = sortByValues(hashmap_PwGivenR);
//        // ---
        //

        // +++ selecting top numFeedbackTerms terms and normalize
        int expansionTermCount = 0;
        float normFactor = 0;

        list_PwGivenR = new ArrayList<>(hashmap_PwGivenR.values());
        hashmap_PwGivenR = new LinkedHashMap<>();
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTerms)
                    break;
            }
            //* else: The t is already there in the hash-map 
        }
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            //wp.p_w_given_R /= normFactor;
            if(normFactor != 0)
                wp.p_w_given_R /= normFactor;
            else
                wp.p_w_given_R = 0;
        }
        // -- Normalizing done

        String[] analyzedQuery = query.queryFieldAnalyze(analyzer, query.qtitle).split("\\s+");
        //------------ ++Ani
//        String[] composedTerms = query.composition.split("\\s+");
//        String[] composedQuery = getUniqTerms((String[]) ArrayUtils.addAll(analyzedQuery, composedTerms)).toArray(new String[0]);
        //------------ --Ani    

        normFactor = 0;
        //* Each w of R: P(w|R) to be (1-QMIX)*P(w|R) 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            String key = entrySet.getKey();
            WordProbability value = entrySet.getValue();
            value.p_w_given_R = value.p_w_given_R * (1.0f-QMIX);
            normFactor += value.p_w_given_R;
        }

        // Now P(w|R) = (1-QMIX)*P(w|R)
        //* Each w which are also query terms: P(w|R) += QMIX*P(w|Q)
        //      P(w|Q) = tf(w,Q)/|Q|
        for (String qTerm : analyzedQuery) { // without word composition
        //for (String qTerm : composedQuery) { // with word composition
            WordProbability oldProba = hashmap_PwGivenR.get(qTerm);
            float newProb = QMIX * returnMLE_of_q_in_Q(analyzedQuery, qTerm); // without word composition
            //float newProb = QMIX * returnMLE_of_q_in_Q(composedQuery, qTerm); // with word composition
            normFactor += newProb;
            if (null != oldProba) { // qTerm is in R
                oldProba.p_w_given_R += newProb;
                hashmap_PwGivenR.put(qTerm, oldProba);
            }
            else  // the qTerm is not in R
                hashmap_PwGivenR.put(qTerm, new WordProbability(qTerm, newProb));
        }

        // ++ Normalizing
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            //wp.p_w_given_R /= normFactor;
            if(normFactor != 0)
                wp.p_w_given_R /= normFactor;
            else
                wp.p_w_given_R = 0;
        }
        // -- Normalizing done

        return hashmap_PwGivenR;
    } // end RM3()
    
    // Modified 'RM1' i.e. Modified top tanked docs. i.e. ToIS term level on top ranked hashMap. P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d) * contextAppro(w)}.
    public HashMap RM1_2(TRECQuery query, TopDocs topDocs) throws Exception {

        float p_W_GivenR_one_doc;

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;
            float contextApproVal = getTermLevelContextualAppropriateness(query, t);
            //float contextApproVal = getTermLevelContextualAppropriateness_singleContextBased(query, t);
            float KDEScore = 1.0f;//getKDEScore(t, query, 1.0f);

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                
                //System.out.println("(" + t + "): " + KDEScore);
                // Ani... ekhane KDERLM dhokate hobe. semantic similarity goon hobe
                p_W_GivenR_one_doc += 
                    return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId) *
                        //((contextApproVal!=0.0f)?contextApproVal:1) *   // ToIS term level (joint) contextual appropriatenes of term 't'
                        contextApproVal *
                        ((KDEScore!=0.0f)?KDEScore:1); // Multiplying KDE socre of term 't' based on 'query' terms
            }
            
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }   // ends RM1()


    // Modified 'RM3' i.e. Modified top tanked docs. i.e. ToIS term level on top ranked hashMap. P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d) * contextAppro(w)}.
    public HashMap RM3_2(TRECQuery query, TopDocs topDocs) throws Exception {

        hashmap_PwGivenR = new LinkedHashMap<>();

        hashmap_PwGivenR = RM1_2(query, topDocs);
        //hashmap_PwGivenR = KDERM1(query, topDocs);
        //hashmap_PwGivenR = KDERM1_2(query, topDocs);
        // hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        
//        // +++ Inserting the idf factor
//        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
//            String key = entrySet.getKey();
//            WordProbability value = entrySet.getValue();
//            value.p_w_given_R *= Math.log(docCount/feedbackTermStats.get(key).getDF()+1);
//            hashmap_PwGivenR.put(key, value);
//        }
//        hashmap_PwGivenR = sortByValues(hashmap_PwGivenR);
//        // ---
        //

        // +++ selecting top numFeedbackTerms terms and normalize
        int expansionTermCount = 0;
        float normFactor = 0;

        list_PwGivenR = new ArrayList<>(hashmap_PwGivenR.values());
        hashmap_PwGivenR = new LinkedHashMap<>();
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTerms)
                    break;
            }
            //* else: The t is already there in the hash-map 
        }
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            //wp.p_w_given_R /= normFactor;
            if(normFactor != 0)
                wp.p_w_given_R /= normFactor;
            else
                wp.p_w_given_R = 0;
        }
        // -- Normalizing done

        String[] analyzedQuery = query.queryFieldAnalyze(analyzer, query.qtitle).split("\\s+");

        normFactor = 0;
        //* Each w of R: P(w|R) to be (1-QMIX)*P(w|R) 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            String key = entrySet.getKey();
            WordProbability value = entrySet.getValue();
            value.p_w_given_R = value.p_w_given_R * (1.0f-QMIX);
            normFactor += value.p_w_given_R;
        }

        // Now P(w|R) = (1-QMIX)*P(w|R)
        //* Each w which are also query terms: P(w|R) += QMIX*P(w|Q)
        //      P(w|Q) = tf(w,Q)/|Q|
        for (String qTerm : analyzedQuery) {
            WordProbability oldProba = hashmap_PwGivenR.get(qTerm);
            float newProb = QMIX * returnMLE_of_q_in_Q(analyzedQuery, qTerm);
            normFactor += newProb;
            if (null != oldProba) { // qTerm is in R
                oldProba.p_w_given_R += newProb;
                hashmap_PwGivenR.put(qTerm, oldProba);
            }
            else  // the qTerm is not in R
                hashmap_PwGivenR.put(qTerm, new WordProbability(qTerm, newProb));
        }

        // ++ Normalizing
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            //wp.p_w_given_R /= normFactor;
            if(normFactor != 0)
                wp.p_w_given_R /= normFactor;
            else
                wp.p_w_given_R = 0;
        }
        // -- Normalizing done

        return hashmap_PwGivenR;
    } // end RM3()
    
    // Modified 'RM1' i.e. Modified top tanked docs. i.e. ToIS term level on top ranked hashMap. P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d) * contextAppro(w)}.
    public HashMap RM1_2General(TRECQuery query, TopDocs topDocs) throws Exception {

        float p_W_GivenR_one_doc;

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;
            float contextApproVal = getTermLevelContextualAppropriatenessGeneral(query, t);
            float KDEScore = 1.0f;//getKDEScore(t, query, 1.0f);

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                
                //System.out.println("(" + t + "): " + KDEScore);
                // Ani... ekhane KDERLM dhokate hobe. semantic similarity goon hobe
                p_W_GivenR_one_doc += 
                    return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId) *
                        ((contextApproVal!=0.0f)?contextApproVal:1) *   // ToIS term level (joint) contextual appropriatenes of term 't'
                        ((KDEScore!=0.0f)?KDEScore:1); // Multiplying KDE socre of term 't' based on 'query' terms
            }
            
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }   // ends RM1()


    // Modified 'RM3' i.e. Modified top tanked docs. i.e. ToIS term level on top ranked hashMap. P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d) * contextAppro(w)}.
    public HashMap RM3_2General(TRECQuery query, TopDocs topDocs) throws Exception {

        hashmap_PwGivenR = new LinkedHashMap<>();

        hashmap_PwGivenR = RM1_2General(query, topDocs);
        //hashmap_PwGivenR = KDERM1_2General(query, topDocs);
        // hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        
//        // +++ Inserting the idf factor
//        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
//            String key = entrySet.getKey();
//            WordProbability value = entrySet.getValue();
//            value.p_w_given_R *= Math.log(docCount/feedbackTermStats.get(key).getDF()+1);
//            hashmap_PwGivenR.put(key, value);
//        }
//        hashmap_PwGivenR = sortByValues(hashmap_PwGivenR);
//        // ---
        //

        // +++ selecting top numFeedbackTerms terms and normalize
        int expansionTermCount = 0;
        float normFactor = 0;

        list_PwGivenR = new ArrayList<>(hashmap_PwGivenR.values());
        hashmap_PwGivenR = new LinkedHashMap<>();
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTerms)
                    break;
            }
            //* else: The t is already there in the hash-map 
        }
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            //wp.p_w_given_R /= normFactor;
            if(normFactor != 0)
                wp.p_w_given_R /= normFactor;
            else
                wp.p_w_given_R = 0;
        }
        // -- Normalizing done

        String[] analyzedQuery = query.queryFieldAnalyze(analyzer, query.qtitle).split("\\s+");

        normFactor = 0;
        //* Each w of R: P(w|R) to be (1-QMIX)*P(w|R) 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            String key = entrySet.getKey();
            WordProbability value = entrySet.getValue();
            value.p_w_given_R = value.p_w_given_R * (1.0f-QMIX);
            normFactor += value.p_w_given_R;
        }

        // Now P(w|R) = (1-QMIX)*P(w|R)
        //* Each w which are also query terms: P(w|R) += QMIX*P(w|Q)
        //      P(w|Q) = tf(w,Q)/|Q|
        for (String qTerm : analyzedQuery) {
            WordProbability oldProba = hashmap_PwGivenR.get(qTerm);
            float newProb = QMIX * returnMLE_of_q_in_Q(analyzedQuery, qTerm);
            normFactor += newProb;
            if (null != oldProba) { // qTerm is in R
                oldProba.p_w_given_R += newProb;
                hashmap_PwGivenR.put(qTerm, oldProba);
            }
            else  // the qTerm is not in R
                hashmap_PwGivenR.put(qTerm, new WordProbability(qTerm, newProb));
        }

        // ++ Normalizing
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            //wp.p_w_given_R /= normFactor;
            if(normFactor != 0)
                wp.p_w_given_R /= normFactor;
            else
                wp.p_w_given_R = 0;
        }
        // -- Normalizing done

        return hashmap_PwGivenR;
    }
    
    // Ani... to use from WebDocSearcher_TRECCS_Novel.java
    public BooleanQuery getExpandedQueryDirect(HashMap<String, WordProbability> expandedQuery, TRECQuery query) throws Exception {

        BooleanQuery booleanQuery = new BooleanQuery();
        
        for (Map.Entry<String, WordProbability> entrySet : expandedQuery.entrySet()) {
            String key = entrySet.getKey();
            if(key.contains(":"))
                continue;
            WordProbability wProba = entrySet.getValue();

            // Ani... term level contextual appropriatenes
            //float contextApproVal = getTermLevelContextualAppropriateness(query, key);

            float value = wProba.p_w_given_R;// + contextApproVal;
            
            String fieldToSearch = this.fieldForFeedback;
            Term t = new Term(fieldToSearch, key);
            Query tq = new TermQuery(t);
            tq.setBoost(value);
            BooleanQuery.setMaxClauseCount(4096);
            booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
        }

        return booleanQuery;
    } // ends getExpandedQuery()

    /**
     * Returns the expanded query in BooleanQuery form with P(w|R) as 
     * corresponding weights for the expanded terms
     * @param expandedQuery The expanded query
     * @param query The query
     * @return BooleanQuery to be used for consequent re-retrieval
     * @throws Exception 
     */
    public BooleanQuery getExpandedQuery(HashMap<String, WordProbability> expandedQuery, TRECQuery query) throws Exception {

        BooleanQuery booleanQuery = new BooleanQuery();
        
        for (Map.Entry<String, WordProbability> entrySet : expandedQuery.entrySet()) {
            String key = entrySet.getKey();
            if(key.contains(":"))
                continue;
            WordProbability wProba = entrySet.getValue();
            float value = wProba.p_w_given_R;

            Term t = new Term(rblm.fieldToSearch, key);
            Query tq = new TermQuery(t);
            tq.setBoost(value);
            BooleanQuery.setMaxClauseCount(4096);
            booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
        }

        return booleanQuery;
    } // ends getExpandedQuery()

    /**
     * Rerank the result depending on the KL-Divergence between the estimated relevance model 
     *  and individual document model.
     * @param hashmap_topM_PwGivenR Top M terms with highest P(w|R)
     * @param query The raw query (unanalyzed)
     * @param topDocs Initial retrieved documents
     * @throws Exception 
     */
    public List<NewScore> rerankUsingRBLM_Doi(HashMap<String, WordProbability> hashmap_topM_PwGivenR, 
        TRECQuery query, TopDocs topDocs) throws Exception {

        List<NewScore> finalList = new ArrayList<>();
        ScoreDoc[] hits;

        int hits_length;
        String w;

        PerTermStat ptsFromDocument;

        hits = topDocs.scoreDocs;
        hits_length = hits.length;               // number of documents retrieved in the first retrieval

        double score;
        double preComputed_p_w_R;
        double singleTerm_p_w_d;

        for (int i = 0; i < hits_length; i++) {
            int luceneDocId = hits[i].doc;
            Document d = indexSearcher.doc(luceneDocId);
            DocumentVector dv = new DocumentVector();
            dv = dv.getDocumentVector(luceneDocId, indexReader);

            score = 0;

            for (Map.Entry<String, WordProbability> entrySet : hashmap_topM_PwGivenR.entrySet()) {
            // for each of the words in top numFeedbackTerms terms in R
                w = entrySet.getKey();
                ptsFromDocument     = dv.docPerTermStat.get(w);
//                ptsFromCollection   = collStat.perTermStat.get(w);
                preComputed_p_w_R = entrySet.getValue().p_w_given_R;

                singleTerm_p_w_d = ( ((ptsFromDocument!=null)?(mixingLambda * (double)ptsFromDocument.getCF() / (double)dv.getDocSize()):(0.0))// );
                    + (1.0f-mixingLambda)*(getCollectionProbability(w, indexReader, fieldForFeedback)));
//                     + ((ptsFromCollection!=null)?((1-mixingLambda)*(double)ptsFromCollection.getCF() / (double)vocSize):(0.0)) );
                score +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));

            } // ends for each t in top numFeedbackTerms terms in R
            
            // Ani... Ei 'score' value-ta holo KLD between i-th doc and 'hashmap_topM_PwGivenR'

            finalList.add(new NewScore(score, d.get(FIELD_ID)));
        } //ends for each initially retrieved documents

        Collections.sort(finalList, new Comparator<NewScore>(){
            @Override
            public int compare(NewScore t, NewScore t1) {
                return t.score>t1.score?1:t.score==t1.score?0:-1;
            }
        });

        return finalList;
    }

    /**
     * Rerank the result depending on the KL-Divergence between the estimated relevance model 
     *  and individual document model.
     * @param hashmap_topM_PwGivenR Top M terms with highest P(w|R)
     * @param query The raw query (unanalyzed)
     * @param topDocs Initial retrieved documents
     * @throws Exception 
     */
    public List<NewScore> rerankUsingRBLM(HashMap<String, WordProbability> hashmap_topM_PwGivenR, 
        TRECQuery query, TopDocs topDocs) throws Exception {

        List<NewScore> finalList = new ArrayList<>();
        ScoreDoc[] hits;

        int hits_length;
        String w;

        PerTermStat ptsFromDocument;

        hits = topDocs.scoreDocs;
        hits_length = hits.length;               // number of documents retrieved in the first retrieval

        double score;
        double preComputed_p_w_R;
        double singleTerm_p_w_d;

        for (int i = 0; i < hits_length; i++) {
            int luceneDocId = hits[i].doc;
            Document d = indexSearcher.doc(luceneDocId);
            DocumentVector dv = new DocumentVector();
            dv = dv.getDocumentVector(luceneDocId, indexReader);

            score = 0;

            for (Map.Entry<String, WordProbability> entrySet : hashmap_topM_PwGivenR.entrySet()) {
            // for each of the words in top numFeedbackTerms terms in R
                w = entrySet.getKey();
                ptsFromDocument     = dv.docPerTermStat.get(w);
//                ptsFromCollection   = collStat.perTermStat.get(w);
                preComputed_p_w_R = entrySet.getValue().p_w_given_R;

		if (ptsFromDocument != null) {
                	//singleTerm_p_w_d = Math.log(1 + mixingLambda/(1-mixingLambda) * ptsFromDocument.getCF()/(double)dv.getDocSize() * reciprocalCollProb(w, indexReader, fieldForFeedback));  // Reciprocal
                        singleTerm_p_w_d = ( ((ptsFromDocument!=null)?(mixingLambda * (double)ptsFromDocument.getCF() / (double)dv.getDocSize()):(0.0)) + (1.0f-mixingLambda)*(getCollectionProbability(w, indexReader, fieldForFeedback)));    // Doi
                        //singleTerm_p_w_d = ptsFromDocument.getCF()/(double)dv.getDocSize(); // DG: no smoothing needed
                        
                        //score +=  (preComputed_p_w_R * (double)Math.log(1 + (preComputed_p_w_R/singleTerm_p_w_d)));
                        score +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));
		}

		//+++DG:
                // singleTerm_p_w_d = ( ((ptsFromDocument!=null)?(mixingLambda * (double)ptsFromDocument.getCF() / (double)dv.getDocSize()):(0.0))// );
                // + (1.0f-mixingLambda)*(getCollectionProbability(w, indexReader, fieldForFeedback)));
		//---DG
		    
//                     + ((ptsFromCollection!=null)?((1-mixingLambda)*(double)ptsFromCollection.getCF() / (double)vocSize):(0.0)) );
                //score +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));

            } // ends for each t in top numFeedbackTerms terms in R
            
            // Ani... Ei 'score' value-ta holo KLD between i-th doc and 'hashmap_topM_PwGivenR'
            finalList.add(new NewScore(score, d.get(FIELD_ID)));
        } //ends for each initially retrieved documents

        Collections.sort(finalList, new Comparator<NewScore>(){
            @Override
            public int compare(NewScore t, NewScore t1) {
                return t.score>t1.score?1:t.score==t1.score?0:-1;
            }
        });

        return finalList;
    }
    
public List<NewScore> rerankUsingRBLM_Neg(HashMap<String, WordProbability> hashmap_topM_PwGivenR, 
        TRECQuery query, TopDocs topDocs) throws Exception {

        List<NewScore> finalList = new ArrayList<>();
        ScoreDoc[] hits;

        int hits_length;
        String w;

        PerTermStat ptsFromDocument;

        hits = topDocs.scoreDocs;
        hits_length = hits.length;               // number of documents retrieved in the first retrieval

        double score;
        double preComputed_p_w_R;
        double singleTerm_p_w_d;

        for (int i = 0; i < hits_length; i++) {
            int luceneDocId = hits[i].doc;
            Document d = indexSearcher.doc(luceneDocId);
            DocumentVector dv = new DocumentVector();
            dv = dv.getDocumentVector(luceneDocId, indexReader);

            score = 0;

            for (Map.Entry<String, WordProbability> entrySet : hashmap_topM_PwGivenR.entrySet()) {
            // for each of the words in top numFeedbackTerms terms in R
                w = entrySet.getKey();
                ptsFromDocument     = dv.docPerTermStat.get(w);
//                ptsFromCollection   = collStat.perTermStat.get(w);
                preComputed_p_w_R = entrySet.getValue().p_w_given_R;

		if (ptsFromDocument != null) {
                	singleTerm_p_w_d = Math.log(1 + mixingLambda/(1-mixingLambda) * ptsFromDocument.getCF()/(double)dv.getDocSize() * reciprocalCollProb(w, indexReader, fieldForFeedback));  // Reciprocal
                        //singleTerm_p_w_d = ( ((ptsFromDocument!=null)?(mixingLambda * (double)ptsFromDocument.getCF() / (double)dv.getDocSize()):(0.0)) + (1.0f-mixingLambda)*(getCollectionProbability(w, indexReader, fieldForFeedback)));    // Doi
                        //singleTerm_p_w_d = ptsFromDocument.getCF()/(double)dv.getDocSize(); // DG: no smoothing needed
                        
                        score +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));
		}

		//+++DG:
                // singleTerm_p_w_d = ( ((ptsFromDocument!=null)?(mixingLambda * (double)ptsFromDocument.getCF() / (double)dv.getDocSize()):(0.0))// );
                // + (1.0f-mixingLambda)*(getCollectionProbability(w, indexReader, fieldForFeedback)));
		//---DG
		    
//                     + ((ptsFromCollection!=null)?((1-mixingLambda)*(double)ptsFromCollection.getCF() / (double)vocSize):(0.0)) );
                //score +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));

            } // ends for each t in top numFeedbackTerms terms in R
            
            // Ani... Ei 'score' value-ta holo KLD between i-th doc and 'hashmap_topM_PwGivenR'
            finalList.add(new NewScore(score, d.get(FIELD_ID)));
        } //ends for each initially retrieved documents

        Collections.sort(finalList, new Comparator<NewScore>(){
            @Override
            public int compare(NewScore t, NewScore t1) {
                return t.score<t1.score?1:t.score==t1.score?0:-1;
            }
        });

        return finalList;
    }

    private static HashMap sortByValues(HashMap map) {
        List<Map.Entry<String, WordProbability>> list = new ArrayList(map.entrySet());
        // Defined Custom Comparator here
        Collections.sort(list, new Comparator<Map.Entry<String, WordProbability>>() {
            @Override
            public int compare(Map.Entry<String, WordProbability> t1, Map.Entry<String, WordProbability> t2) {
                return t1.getValue().p_w_given_R<t2.getValue().p_w_given_R?1:t1.getValue().p_w_given_R==t2.getValue().p_w_given_R?0:-1;
            }
        });

        // Copying the sorted list in HashMap
        // using LinkedHashMap to preserve the insertion order
        HashMap sortedHashMap = new LinkedHashMap();
        for (Map.Entry entry : list) {
            sortedHashMap.put(entry.getKey(), entry.getValue());
        }
        return sortedHashMap;
    }
    
    public List<NewScore> rerankUsingRBLM_PosNeg(HashMap<String, WordProbability> hashmap_topM_PwGivenR_Pos, 
        HashMap<String, WordProbability> hashmap_topM_PwGivenR_Neg, TRECQuery query, TopDocs topDocs) throws Exception {

        List<NewScore> finalList = new ArrayList<>();
        ScoreDoc[] hits;

        int hits_length;
        String w;

        PerTermStat ptsFromDocument;

        hits = topDocs.scoreDocs;
        hits_length = hits.length;               // number of documents retrieved in the first retrieval

        double score_Pos, score_Neg;
        double preComputed_p_w_R;
        double singleTerm_p_w_d;

        for (int i = 0; i < hits_length; i++) {
            int luceneDocId = hits[i].doc;
            Document d = indexSearcher.doc(luceneDocId);
            DocumentVector dv = new DocumentVector();
            dv = dv.getDocumentVector(luceneDocId, indexReader);

            score_Pos = 0;

            for (Map.Entry<String, WordProbability> entrySet : hashmap_topM_PwGivenR_Pos.entrySet()) {
            // for each of the words in top numFeedbackTerms terms in R
                w = entrySet.getKey();
                ptsFromDocument     = dv.docPerTermStat.get(w);
//                ptsFromCollection   = collStat.perTermStat.get(w);
                preComputed_p_w_R = entrySet.getValue().p_w_given_R;

                // DG calculation
                if (ptsFromDocument != null) {
                    singleTerm_p_w_d = Math.log(1 + mixingLambda/(1-mixingLambda) * ptsFromDocument.getCF()/(double)dv.getDocSize() * reciprocalCollProb(w, indexReader, fieldForFeedback));  // Reciprocal
                    //singleTerm_p_w_d = (((ptsFromDocument != null) ? (mixingLambda * (double) ptsFromDocument.getCF() / (double) dv.getDocSize()) : (0.0)) + (1.0f - mixingLambda) * (getCollectionProbability(w, indexReader, fieldForFeedback)));    // Doi
                    //singleTerm_p_w_d = ptsFromDocument.getCF()/(double)dv.getDocSize(); // DG: no smoothing needed

                    score_Pos +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));
                }
                // Doi calculation
//                singleTerm_p_w_d = ( ((ptsFromDocument!=null)?(mixingLambda * (double)ptsFromDocument.getCF() / (double)dv.getDocSize()):(0.0))// );
//                    + (1.0f-mixingLambda)*(getCollectionProbability(w, indexReader, fieldForFeedback)));
////                     + ((ptsFromCollection!=null)?((1-mixingLambda)*(double)ptsFromCollection.getCF() / (double)vocSize):(0.0)) );
//                score_Pos +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));

            } // ends for each t in top numFeedbackTerms terms in R
            // 'score_Pos' is the KLD between i-th doc and 'hashmap_topM_PwGivenR_Pos'
            
            score_Neg = 0;
            if(hashmap_topM_PwGivenR_Neg.isEmpty() == false) {
                for (Map.Entry<String, WordProbability> entrySet : hashmap_topM_PwGivenR_Neg.entrySet()) {
                // for each of the words in top numFeedbackTerms terms in R
                    w = entrySet.getKey();
                    ptsFromDocument     = dv.docPerTermStat.get(w);
    //                ptsFromCollection   = collStat.perTermStat.get(w);
                    preComputed_p_w_R = entrySet.getValue().p_w_given_R;
                    
                    // DG calculation
                    if (ptsFromDocument != null) {
                        singleTerm_p_w_d = Math.log(1 + mixingLambda/(1-mixingLambda) * ptsFromDocument.getCF()/(double)dv.getDocSize() * reciprocalCollProb(w, indexReader, fieldForFeedback));  // Reciprocal
                        //singleTerm_p_w_d = (((ptsFromDocument != null) ? (mixingLambda * (double) ptsFromDocument.getCF() / (double) dv.getDocSize()) : (0.0)) + (1.0f - mixingLambda) * (getCollectionProbability(w, indexReader, fieldForFeedback)));    // Doi
                        //singleTerm_p_w_d = ptsFromDocument.getCF()/(double)dv.getDocSize(); // DG: no smoothing needed

                        score_Neg +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));
                    }

//                    // Doi calculation
//                    singleTerm_p_w_d = ( ((ptsFromDocument!=null)?(mixingLambda * (double)ptsFromDocument.getCF() / (double)dv.getDocSize()):(0.0))// );
//                        + (1.0f-mixingLambda)*(getCollectionProbability(w, indexReader, fieldForFeedback)));
////                         + ((ptsFromCollection!=null)?((1-mixingLambda)*(double)ptsFromCollection.getCF() / (double)vocSize):(0.0)) );
//                    score_Neg +=  (preComputed_p_w_R * (double)Math.log(preComputed_p_w_R/singleTerm_p_w_d));

                } // ends for each t in top numFeedbackTerms terms in R
                // 'score_Neg' is the KLD between i-th doc and 'hashmap_topM_PwGivenR_Neg'
                score_Pos = score_Pos/score_Neg;
            }


            finalList.add(new NewScore(score_Pos, d.get(FIELD_ID)));
        } //ends for each initially retrieved documents

        Collections.sort(finalList, new Comparator<NewScore>(){
            @Override
            public int compare(NewScore t, NewScore t1) {
                return t.score>t1.score?1:t.score==t1.score?0:-1;
            }
        });

        return finalList;
    }

    // Standard RM1 with terms boosted by the rating of the document (only for rated user history)
    public HashMap RM1Customized(TRECQuery query, TopDocs topDocs, UserPreference userPref) throws Exception {
        
        float p_W_GivenR_one_doc;

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;
            float contextApproVal = 1.0f; //getTermLevelContextualAppropriateness(query, t);

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                // Ani... ekhane KDERLM dhokate hobe. semantic similarity goon hobe
                String docID = indexSearcher.doc(luceneDocId).get(FIELD_ID);
                float KDEScore = 1.0f; //getKDEScore(t, query, 1.0f);
                p_W_GivenR_one_doc += 
                    return_Smoothed_MLE_Log(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId) *
                        ((getDocumentRating(userPref, docID)!=-99)?getDocumentRating(userPref, docID):1) * // multiplying doc rating
                        //((getDocumentRating(userPref, docID)!=-99 && return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)) != 1.0f)?getDocumentRating(userPref, docID):1) * // multiplying doc rating
                        ((contextApproVal!=0.0f)?contextApproVal:1) *   // ToIS term level (joint) contextual appropriatenes of term 't'
                            ((KDEScore!=0.0f)?KDEScore:1); // Multiplying KDE socre of term 't' based on 'query' terms
            }
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }   // ends RM1()

    // Standard RM3 with terms boosted by the rating of the document (only for rated user history)
    public HashMap RM3Customized(TRECQuery query, TopDocs topDocs, UserPreference userPref) throws Exception {

        hashmap_PwGivenR = new LinkedHashMap<>();

        hashmap_PwGivenR = RM1Customized(query, topDocs, userPref);
        //hashmap_PwGivenR = KDERM1Customized(query, topDocs, userPref);
        // hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        
//        // +++ Inserting the idf factor
//        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
//            String key = entrySet.getKey();
//            WordProbability value = entrySet.getValue();
//            value.p_w_given_R *= Math.log(docCount/feedbackTermStats.get(key).getDF()+1);
//            hashmap_PwGivenR.put(key, value);
//        }
//        hashmap_PwGivenR = sortByValues(hashmap_PwGivenR);
//        // ---
        //

        // +++ selecting top numFeedbackTerms terms and normalize
        int expansionTermCount = 0;
        float normFactor = 0;

        list_PwGivenR = new ArrayList<>(hashmap_PwGivenR.values());
        hashmap_PwGivenR = new LinkedHashMap<>();
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTerms)
                    break;
            }
            //* else: The t is already there in the hash-map 
        }
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
        }
        // -- Normalizing done

        String[] analyzedQuery = query.queryFieldAnalyze(analyzer, query.qtitle).split("\\s+");

        normFactor = 0;
        //* Each w of R: P(w|R) to be (1-QMIX)*P(w|R) 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            String key = entrySet.getKey();
            WordProbability value = entrySet.getValue();
            value.p_w_given_R = value.p_w_given_R * (1.0f-QMIX);
            normFactor += value.p_w_given_R;
        }

        // Now P(w|R) = (1-QMIX)*P(w|R)
        //* Each w which are also query terms: P(w|R) += QMIX*P(w|Q)
        //      P(w|Q) = tf(w,Q)/|Q|
        for (String qTerm : analyzedQuery) {
            WordProbability oldProba = hashmap_PwGivenR.get(qTerm);
            float newProb = QMIX * returnMLE_of_q_in_Q(analyzedQuery, qTerm);
            normFactor += newProb;
            if (null != oldProba) { // qTerm is in R
                oldProba.p_w_given_R += newProb;
                hashmap_PwGivenR.put(qTerm, oldProba);
            }
            else  // the qTerm is not in R
                hashmap_PwGivenR.put(qTerm, new WordProbability(qTerm, newProb));
        }

        // ++ Normalizing
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
        }
        // -- Normalizing done

        return hashmap_PwGivenR;
    } // end RM3()
    
    // Modified 'RM1Customized' i.e. ModifiedHistory. i.e. ToIS term level on history hashMap. P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d) * rating(d) * contextAppro(w)}.
    public HashMap RM1Customized2(TRECQuery query, TopDocs topDocs, UserPreference userPref) throws Exception {
        
        float p_W_GivenR_one_doc;

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;
            float contextApproVal = getTermLevelContextualAppropriateness(query, t);
            //float contextApproVal = getTermLevelContextualAppropriateness_singleContextBased(query, t);

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                // Ani... ekhane KDERLM dhokate hobe. semantic similarity goon hobe
                String docID = indexSearcher.doc(luceneDocId).get(FIELD_ID);
                float KDEScore = 1.0f; //getKDEScore(t, query, 1.0f);
                p_W_GivenR_one_doc += 
                    return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId) *
                        ((getDocumentRating(userPref, docID)!=-99)?getDocumentRating(userPref, docID):1) * // multiplying doc rating
                        //((getDocumentRating(userPref, docID)!=-99 && return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)) != 1.0f)?getDocumentRating(userPref, docID):1) * // multiplying doc rating
                        //((contextApproVal!=0.0f)?contextApproVal:1) *   // ToIS term level (joint) contextual appropriatenes of term 't'
                        contextApproVal *   // singleContext based [0, +1]
                            ((KDEScore!=0.0f)?KDEScore:1); // Multiplying KDE socre of term 't' based on 'query' terms
            }
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }   // ends RM1()

    // Modified 'RM3Customized' i.e. ModifiedHistory. i.e. ToIS term level on history hashMap. P(w|R) = \sum{d\in PRD} {smoothedMLE(w,d)*smoothedMLE(Q,d) * rating(d) * contextAppro(w)}.
    public HashMap RM3Customized2(TRECQuery query, TopDocs topDocs, UserPreference userPref) throws Exception {

        hashmap_PwGivenR = new LinkedHashMap<>();

        hashmap_PwGivenR = RM1Customized2(query, topDocs, userPref);
        //hashmap_PwGivenR = KDERM1Customized(query, topDocs, userPref);
        // hashmap_PwGivenR has all terms of PRDs along with their probabilities 

        
//        // +++ Inserting the idf factor
//        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
//            String key = entrySet.getKey();
//            WordProbability value = entrySet.getValue();
//            value.p_w_given_R *= Math.log(docCount/feedbackTermStats.get(key).getDF()+1);
//            hashmap_PwGivenR.put(key, value);
//        }
//        hashmap_PwGivenR = sortByValues(hashmap_PwGivenR);
//        // ---
        //

        // +++ selecting top numFeedbackTerms terms and normalize
        int expansionTermCount = 0;
        float normFactor = 0;

        list_PwGivenR = new ArrayList<>(hashmap_PwGivenR.values());
        hashmap_PwGivenR = new LinkedHashMap<>();
        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
                expansionTermCount++;
                normFactor += singleTerm.p_w_given_R;
                if(expansionTermCount>=numFeedbackTerms)
                    break;
            }
            //* else: The t is already there in the hash-map 
        }
        // ++ Normalizing 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
        }
        // -- Normalizing done

        String[] analyzedQuery = query.queryFieldAnalyze(analyzer, query.qtitle).split("\\s+");

        normFactor = 0;
        //* Each w of R: P(w|R) to be (1-QMIX)*P(w|R) 
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            String key = entrySet.getKey();
            WordProbability value = entrySet.getValue();
            value.p_w_given_R = value.p_w_given_R * (1.0f-QMIX);
            normFactor += value.p_w_given_R;
        }

        // Now P(w|R) = (1-QMIX)*P(w|R)
        //* Each w which are also query terms: P(w|R) += QMIX*P(w|Q)
        //      P(w|Q) = tf(w,Q)/|Q|
        for (String qTerm : analyzedQuery) {
            WordProbability oldProba = hashmap_PwGivenR.get(qTerm);
            float newProb = QMIX * returnMLE_of_q_in_Q(analyzedQuery, qTerm);
            normFactor += newProb;
            if (null != oldProba) { // qTerm is in R
                oldProba.p_w_given_R += newProb;
                hashmap_PwGivenR.put(qTerm, oldProba);
            }
            else  // the qTerm is not in R
                hashmap_PwGivenR.put(qTerm, new WordProbability(qTerm, newProb));
        }

        // ++ Normalizing
        for (Map.Entry<String, WordProbability> entrySet : hashmap_PwGivenR.entrySet()) {
            WordProbability wp = entrySet.getValue();
            wp.p_w_given_R /= normFactor;
        }
        // -- Normalizing done

        return hashmap_PwGivenR;
    } // end RM3()
    
    // Returns the doc rating (user history) if the doc exists in the history. Rating could be -2, -1, 0, +1, +2, +3, +4
    // Returns -99, otherwise
    public int getDocumentRating(UserPreference userPref, String docID) throws Exception {
//userPref.get(userPrefIndex)
        for (int i = 0; i < userPref.nPreference; ++i) {
            if(userPref.docId[i].equals(docID))
                return userPref.rating[i];
        }
        return -99;
    }
    
    public class cmpW2VModel implements Comparator<String[]> {

        @Override
        public int compare(String a[], String b[]) {
            return a[0].compareTo(b[0]) > 0 ? 1 : a[0].compareTo(b[0]) == 0 ? 0 : -1;   // standard sort (ascending order)
            //return a.score<b.score?1:a.score==b.score?0:-1; // reverse order
        }
    }
    
    public class cmpW2V implements Comparator<Word2vec> {
        @Override
        public int compare (Word2vec a, Word2vec b) {
            return a.term.compareTo(b.term)>0?1:a.term.compareTo(b.term)==0?0:-1;
            //return a.consineScore>b.consineScore?1:a.consineScore==b.consineScore?0:-1;   // standard sort (ascending order)
            //return a.consineScore<b.consineScore?1:a.consineScore==b.consineScore?0:-1; // reverse order
        }
    }
    
    public class cmpBERTVec implements Comparator<BERTVec> {
        @Override
        public int compare (BERTVec a, BERTVec b) {
            return a.term.compareTo(b.term)>0?1:a.term.compareTo(b.term)==0?0:-1;
            //return a.consineScore>b.consineScore?1:a.consineScore==b.consineScore?0:-1;   // standard sort (ascending order)
            //return a.consineScore<b.consineScore?1:a.consineScore==b.consineScore?0:-1; // reverse order
        }
    }
    
    // Get BERT vector of the term 'term' for query 'qID', in 'bert' model
    public float[] getBERTvec (String term, String qID) throws Exception {
        float[] blank = new float[1];
        blank[0] = -999.0f;
        int nBert = bert.size();
        for (int i = 0; i < nBert; ++i) {
            if(qID.equals(bert.get(i).qID)) {
                BERTVec key = new BERTVec();
                key.term = term;
                int index = Collections.binarySearch(bert.get(i).bertVec, key, new cmpBERTVec());
                if(index >= 0)
                    return bert.get(i).bertVec.get(index).vector;
                else
                    return blank;
            }
        }
        return blank;
    }
    
    public int getTermIndex (String term) throws Exception {    // Get index of 'term' in W2V model
        String[] keyTermVector = new String[201];
        keyTermVector[0] = term;
        return Collections.binarySearch(W2Vmodel, keyTermVector, new cmpW2VModel());
    }
    
    public int getW2VTermIndex (String term) throws Exception {    // Get index of 'term' in W2V model
        Word2vec key = new Word2vec();
        key.term = term;
        return Collections.binarySearch(W2V, key, new cmpW2V());
    }
    
    // Returns KDE score of term 'term' based on 'query' terms.
    public float getKDEScore(String term, TRECQuery query, float sigma) throws Exception {

        int index, n;
        float score = gaussianKernel(100, sigma);
        float h;
        String fieldToSearch = this.fieldForFeedback;
        String[] termsRaw = trecQueryparser.getAnalyzedQuery(query, 1).toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");

        List<String> terms = getUniqTerms(termsRaw);
        List<Word2vec> observedTerms = new ArrayList<>();
        List<Word2vec> candidateTerms = new ArrayList<>();
        
        int nTerms = terms.size();
        for (int i = 0; i < nTerms; ++i) {
            //index = getTermIndex(terms.get(i));
            index = getW2VTermIndex(terms.get(i));
            if(index >= 0) {
                Word2vec temp = new Word2vec();
                temp.term = terms.get(i);
                //temp.vector = convertVectorStringToFloat(W2Vmodel.get(index));
                temp.vector = W2V.get(index).vector;
                observedTerms.add(temp);
            }
        }
        
        int nObservedTerms = observedTerms.size();

        float[] weightArray = new float[nObservedTerms]; // weight array for KDE. Weights for observed terms
        for (int i = 0; i < nObservedTerms; ++i) {
            //weightArray[i] = 1.0f;  // 1.0 means equal weights. Try using tf, Cf, collection probability, tf*IDF etc.
            //weightArray[i] = (float) (getCF(observedTerms.get(i).term) * getIdf(observedTerms.get(i).term));
            //weightArray[i] = (float) (getCF(observedTerms.get(i).term));
            //weightArray[i] = (float) (getCFNormalized(observedTerms.get(i).term));
            
            weightArray[i] = 1.0f;
        }
        
        //index = getTermIndex(term);
        index = getW2VTermIndex(term);
        if(index >= 0) {
            Word2vec temp = new Word2vec();
            temp.term = term;
            //temp.vector = convertVectorStringToFloat(W2Vmodel.get(index));
            temp.vector = W2V.get(index).vector;
            
            //return KDEScoreForTermSelect(temp.vector, observedTerms, weightArray, observedTerms.size(), 1, 1.0f);  // using sigma=1.0, h=1
            n = observedTerms.size();
            h = (float) Math.pow(((4*Math.pow(sigma, 5))/(3*n)), -(1/5));
            //h = 1.0f;
            //System.out.println("||||||||||| KDEScore(" + term + "): " + KDEScoreForTermSelect(temp.vector, observedTerms, weightArray, observedTerms.size(), 1, sigma) + "\th: " + h + "\tn: " + n);
            //return KDEScoreForTermSelect(temp.vector, observedTerms, weightArray, observedTerms.size(), h, sigma);
            return getKDE(temp.vector, observedTerms, weightArray, observedTerms.size(), h, sigma);
        }

        //assert score != 0;

        return score;
    }
    
    public float getKDEScoreBERT(String term, TRECQuery query, float sigma) throws Exception {

        int index, n;
        float score = gaussianKernel(100, sigma);
        float h;
        String fieldToSearch = this.fieldForFeedback;
        String[] termsRaw = trecQueryparser.getAnalyzedQuery(query, 1).toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");

        List<String> terms = getUniqTerms(termsRaw);
        List<Word2vec> observedTerms = new ArrayList<>();
        List<Word2vec> candidateTerms = new ArrayList<>();
        
        int nTerms = terms.size();
        for (int i = 0; i < nTerms; ++i) {
            float[] qVec = getBERTvec(terms.get(i), query.qid);
            if(qVec[0] != -999.0f) {
                Word2vec temp = new Word2vec();
                temp.term = terms.get(i);
                //temp.vector = convertVectorStringToFloat(W2Vmodel.get(index));
                temp.vector = qVec;
                observedTerms.add(temp);
            }
        }
        
        int nObservedTerms = observedTerms.size();

        float[] weightArray = new float[nObservedTerms]; // weight array for KDE. Weights for observed terms
        for (int i = 0; i < nObservedTerms; ++i) {
            //weightArray[i] = 1.0f;  // 1.0 means equal weights. Try using tf, Cf, collection probability, tf*IDF etc.
            //weightArray[i] = (float) (getCF(observedTerms.get(i).term) * getIdf(observedTerms.get(i).term));
            //weightArray[i] = (float) (getCF(observedTerms.get(i).term));
            //weightArray[i] = (float) (getCFNormalized(observedTerms.get(i).term));
            
            weightArray[i] = 1.0f;
        }
        
        //index = getTermIndex(term);
        float[] wVec = getBERTvec(term, query.qid);
        if(wVec[0] != -999.0f) {
            Word2vec temp = new Word2vec();
            temp.term = term;
            //temp.vector = convertVectorStringToFloat(W2Vmodel.get(index));
            temp.vector = wVec;
            
            //return KDEScoreForTermSelect(temp.vector, observedTerms, weightArray, observedTerms.size(), 1, 1.0f);  // using sigma=1.0, h=1
            n = observedTerms.size();
            h = (float) Math.pow(((4*Math.pow(sigma, 5))/(3*n)), -(1/5));
            //System.out.println("||||||||||| KDEScore(" + term + "): " + KDEScoreForTermSelect(temp.vector, observedTerms, weightArray, observedTerms.size(), 1, sigma) + "\th: " + h + "\tn: " + n);
            //return KDEScoreForTermSelect(temp.vector, observedTerms, weightArray, observedTerms.size(), h, sigma);
            return getKDE(temp.vector, observedTerms, weightArray, observedTerms.size(), h, sigma);
        }

        //assert score != 0;

        return score;
    }
    
    // DG
    public float getKDE(float[] x, List<Word2vec> xArray, float[] wArray, int n, float h, float sigma) throws Exception {
        float score = 0.0f;
        
        for (int i = 0; i < n; ++i) {
            //float dist = cosineDistance(x, xArray.get(i).vector);
            //float dist = euclideanDistance(x, xArray.get(i).vector);
            //float dist = euclideanDistance(x, xArray.get(i).vector)/l2norm(x);
            float dist = euclideanDistance(normalizeVec(x), normalizeVec(xArray.get(i).vector));
            score += wArray[i] * gaussianKernel(dist/h, sigma);
        }
        score /= (n * h);
        
        return score;
    }
    
    public float gaussianKernel(float x, float sigma) throws Exception {
        return (float) Math.exp(-Math.pow(x/sigma, 2));
    }
    
    // Returns estimated KDE score of term 'x' (vector representation of x) based on x_i terms ('xArray') where i=0, 1, ..., n-1
    // f_w(x) = 1/nh ∑w_i . K((x - x_i) / h) for i=1, 2, ..., n [weighted KDE with gaussian kernel function K(.), bandwidth h]
    public float KDEScoreForTermSelect(float[] x, List<Word2vec> xArray, float[] wArray, int n, float h, float sigma) throws Exception {
        float score = 0.0f, expPart;

        for (int i = 0; i < n; ++i) {
            //expPart = (float) Math.exp(-( Math.pow(cosineSimilarity(x, xArray.get(i).vector), 2) / 2 * Math.pow(sigma, 2) * Math.pow(h, 2)));
            expPart = (float) Math.exp(-( Math.pow(euclideanDistance(x, xArray.get(i).vector), 2) / 2 * Math.pow(sigma, 2) * Math.pow(h, 2)));
            score += wArray[i] / (Math.sqrt(2 * Math.PI) * sigma) * expPart;
        }
        score /= n * h;
        
        return score;
    }
    
    // Merge two vectors
    public float[] mergeVecs(float[] a, float[] b) throws Exception {
        int n = a.length + b.length;
        float[] c = new float[n];
        int j = 0;
        for (int i = 0; i < a.length; ++i)
            c[j++] = a[i];
        for (int i = 0; i < b.length; ++i)
            c[j++] = b[i];
        return c;
    }
    
    // W2V and RoBERTa vectors 200 + 768 dim
    public float getEuclideanDistance_W2V_and_BERT(String a, String b, TRECQuery query) throws Exception {
        int index_a = getW2VTermIndex(a);
        int index_b = getW2VTermIndex(b);
        float[] aVec_BERT = getBERTvec(a, query.qid);
        float[] bVec_BERT = getBERTvec(b, query.qid);
        if((index_a >= 0) && (aVec_BERT[0] != -999.0f)) {
            float[] aVec_W2V = W2V.get(index_a).vector;
            if((index_b >= 0) && (bVec_BERT[0] != -999.0f)) {
                float[] bVec_W2V = W2V.get(index_b).vector;
                float[] aVec = mergeVecs(aVec_W2V, aVec_BERT);
                //float[] aVec = mergeVecs(normalizeVec(aVec_W2V), normalizeVec(aVec_BERT));
                float[] bVec = mergeVecs(bVec_W2V, bVec_BERT);
                //float[] bVec = mergeVecs(normalizeVec(bVec_W2V), normalizeVec(bVec_BERT));
                //return euclideanDistance(aVec, bVec);
                return euclideanDistance(normalizeVec(aVec), normalizeVec(bVec));
            }
            else if(index_b >= 0) {
                float[] bVec_W2V = W2V.get(index_b).vector;
                return euclideanDistance(normalizeVec(aVec_W2V), normalizeVec(bVec_W2V));
            }
            else if(bVec_BERT[0] != -999.0f)
                return euclideanDistance(normalizeVec(aVec_BERT), normalizeVec(bVec_BERT));
            else
                return 100.0f;
        }
        else if(index_a >= 0) {
            if(index_b >= 0) {
                float[] aVec_W2V = W2V.get(index_a).vector;
                float[] bVec_W2V = W2V.get(index_b).vector;
                return euclideanDistance(normalizeVec(aVec_W2V), normalizeVec(bVec_W2V));
            }
            return 100.0f;
        }
        else if(aVec_BERT[0] != -999.0f) {
            if(bVec_BERT[0] != -999.0f)
                return euclideanDistance(normalizeVec(aVec_BERT), normalizeVec(bVec_BERT));
            else
                return 100.0f;
        }
        else
            return 100.0f;
    }
    
    // RoBERTa vectors 768 dim
    public float getEuclideanDistance_BERT(String a, String b, TRECQuery query) throws Exception {
        float[] aVec = getBERTvec(a, query.qid);
        if(aVec[0] != -999.0f) {
            float[] bVec = getBERTvec(b, query.qid);
            if(bVec[0] != -999.0f)
                return euclideanDistance(normalizeVec(aVec), normalizeVec(bVec));
                //return euclideanDistance(aVec, bVec);
            else
                //return getEuclideanDistance(a, b);
                return 100.0f;
        }
        else
            //return getEuclideanDistance(a, b);
            return 100.0f;
    }
    
    // W2V vectors 200 dim
    public float getEuclideanDistance(String a, String b) throws Exception {
        int index_a = getW2VTermIndex(a);
        int index_b = getW2VTermIndex(b);
        if(index_a >= 0) {
            float[] aVec = W2V.get(index_a).vector;
            if(index_b >= 0) {
                float[] bVec = W2V.get(index_b).vector;
                return euclideanDistance(normalizeVec(aVec), normalizeVec(bVec));
            }
            else
                return 100.0f;
        }
        else
            return 100.0f;
    }
    
    // Returns Euclidean distance between two vectors
    public float euclideanDistance(float[] a, float[] b) throws Exception {
        float sum = 0.0f;

        for (int i = 0; i < a.length; ++i) {
            sum += Math.pow((a[i] - b[i]), 2);
        }
        return (float) Math.sqrt(sum);
    }
    
    // Returns l2norm (length) of vector 'a'
    public float l2norm(float[] a) throws Exception {
        float sum = 0.0f;

        for (int i = 0; i < a.length; ++i) {
            sum += Math.pow(a[i], 2);
        }
        return (float) Math.sqrt(sum);
    }
        
    // Returns cosine distance between two vectors
    public float cosineDistance(float[] a, float[] b) throws Exception {
//        System.out.println("CosSim: " + cosineSimilarity(a, b));
//        System.out.println("CosDis: " + Math.acos(cosineSimilarity(a, b)));
//        System.out.println("EucldDis: " + euclideanDistance(a, b));
//        System.out.println("l2norm: " + l2norm(a));
//        System.out.println("EucldDisNorm: " + euclideanDistance(a, b)/l2norm(a));
//        System.out.println("EucldDisNorm1: " + euclideanDistance(normalizeVec(a), normalizeVec(b)));
//        System.out.println("--------------------------------------------");
//        System.exit(1);
        return (float) Math.acos(cosineSimilarity(a, b));
    }
    
    // Returns cosine similarity between two vectors
    public float cosineSimilarity(float[] a, float[] b) throws Exception {
        float sum = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

        for (int i = 0; i < a.length; ++i) {
            sum += a[i] * b[i];
            sum1 += Math.pow(a[i], 2);
            sum2 += Math.pow(b[i], 2);
        }
        sum /= (Math.sqrt(sum1) * Math.sqrt(sum2));
        return sum;
    }
    
    public List<String> getUniqTerms(String[] termsRaw) throws Exception {
        List<String> terms = new ArrayList<>();
        
        Arrays.sort(termsRaw);
        
        String term = termsRaw[0];
        for (int i = 1; i < termsRaw.length; ++i) {
            if(!term.equals(termsRaw[i])) {
                terms.add(term);
                term = termsRaw[i];
            }
        }
        terms.add(term);

        return terms;
    }
    
    public float[] convertVectorStringToFloat(String[] vector) throws Exception {
        float[] vectorFloat = new float[vector.length];
        for (int i = 1; i < vector.length; ++i) {
            vectorFloat[i - 1] = Float.parseFloat(vector[i]);
        }
        return vectorFloat;
    }
    
    // Returns normalized vector (l2norm) 'a'
    public float[] normalizeVec(float[] a) throws Exception {
        float[] b = new float[a.length];
        float length = l2norm(a);

        for (int i = 0; i < a.length; ++i) {
            b[i] = a[i] / length;
        }
        return b;
    }
    
    public HashMap KDERM1Customized(TRECQuery query, TopDocs topDocs, UserPreference userPref) throws Exception {
        //((getDocumentRating(userPref, docID)!=-99)?getDocumentRating(userPref, docID):1) * // multiplying doc rating
        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            
            String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
            String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");
            
            float contextApproVal = getTermLevelContextualAppropriateness(query, t);
            //float contextApproVal = getTermLevelContextualAppropriateness_singleContextBased(query, t);
            
            float P_wGivenM = 0.0f, KDERLMScore = 0.0f;
            //float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;

            for (String qTerm : analyzedQuery) {

                float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;
                for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
                // for each doc in RF-set
                    int luceneDocId = docEntrySet.getKey();
                    DocumentVector docV = docEntrySet.getValue();
                    String docID = indexSearcher.doc(luceneDocId).get(FIELD_ID);
                    int docRating = ((getDocumentRating(userPref, docID)!=-99)?getDocumentRating(userPref, docID):1);

                    P_wGivenM += return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)) * docRating; // P(w|D) * rating(D)
                    P_qiGivenM += return_Smoothed_MLE(qTerm, docV) * hash_P_Q_Given_D.get(luceneDocId); // P(q_i|D) * P(Q|D)
                    P_QGivenM += hash_P_Q_Given_D.get(luceneDocId);
                }
                
                KDERLMScore += P_wGivenM * // P(w|M)
                                P_qiGivenM * // P(q_i|M)
                                //P_QGivenM * // P(Q|M)
                                contextApproVal *
                                getKDEScoreForSingleQTerm(t, qTerm, 1.0f); // KDE score of 't' based on 'qTerm'
            }
            
            list_PwGivenR.add(new WordProbability(t, KDERLMScore));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    public HashMap KDERM1(TRECQuery query, TopDocs topDocs) throws Exception {

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            
            String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
            String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");
            
            float P_wGivenM = 0.0f, KDERLMScore = 0.0f;
            //float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;

            for (String qTerm : analyzedQuery) {

                float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;
                for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
                // for each doc in RF-set
                    int luceneDocId = docEntrySet.getKey();
                    DocumentVector docV = docEntrySet.getValue();
                    
                    P_wGivenM += return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)); // P(w|D)
                    P_qiGivenM += return_Smoothed_MLE(qTerm, docV) * hash_P_Q_Given_D.get(luceneDocId); // P(q_i|D) * P(Q|D)
                    P_QGivenM += hash_P_Q_Given_D.get(luceneDocId);
                }
                
                KDERLMScore += P_wGivenM * // P(w|M)
                                P_qiGivenM * // P(q_i|M)
                                //P_QGivenM * // P(Q|M)
                                getKDEScoreForSingleQTerm(t, qTerm, 1.0f); // KDE score of 't' based on 'qTerm'
            }
            
            list_PwGivenR.add(new WordProbability(t, KDERLMScore));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    // Not completed
    public HashMap KDERM1_DG(TRECQuery query, TopDocs topDocs) throws Exception {

        float p_W_GivenR_one_doc;

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();
        
        String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
        String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            p_W_GivenR_one_doc = 0;
            //float KDEScore = 1.0f;
            //float KDEScore = getKDEScore(t, query, 1.0f);
            float KDEScore = getKDEScoreBERT(t, query, 1.0f);

            for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set
                int luceneDocId = docEntrySet.getKey();
                float docScore = docEntrySet.getValue().getDocScore(); // Ani
                
                //for (String qTerm : analyzedQuery) {
                
                //System.out.println("(" + t + "): " + KDEScore);
                // Ani... ekhane KDERLM dhokate hobe. semantic similarity goon hobe
                p_W_GivenR_one_doc += 
                    //docScore * // mostly for 2D KDE (experimental)
                    return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)) *
                    hash_P_Q_Given_D.get(luceneDocId) *
                        ((KDEScore!=0.0f)?KDEScore:1); // Multiplying KDE socre of term 't' based on 'query' terms
            }
            
            list_PwGivenR.add(new WordProbability(t, p_W_GivenR_one_doc));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    public HashMap KDERM1_Doi(TRECQuery query, TopDocs topDocs) throws Exception {

        float sigma = 10.0f;
        
        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            
            String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
            String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split("\\s+");
//            String[] composedTerms = query.composition.split("\\s+");
//            String[] composedQuery = getUniqTerms((String[]) ArrayUtils.addAll(analyzedQuery, composedTerms)).toArray(new String[0]);
            
            float P_wGivenM = get_P_wGivenM(t);
            float KDERLMScore = 0.0f;

            for (String qTerm : analyzedQuery) { // without word composition
            //for (String qTerm : composedQuery) { // with word composition

                float P_qiGivenM = get_P_wGivenM(qTerm);
                
                KDERLMScore += P_wGivenM * // P(w|M)
                                P_qiGivenM * // P(q_i|M)
                                //P_QGivenM * // P(Q|M)
                                getKDEScoreForSingleQTermGeneral(t, qTerm, sigma); // KDE score of 't' based on 'qTerm' (W2V)
                                //getKDEScoreForSingleQTermGeneral_BERT(t, qTerm, sigma, query); // KDE score of 't' based on 'qTerm' (BERT)
            }
            
            // avg DG
            //KDERLMScore /= (float) analyzedQuery.length;
            
            list_PwGivenR.add(new WordProbability(t, KDERLMScore));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    public HashMap KDERM1_2D_Doi(TRECQuery query, TopDocs topDocs) throws Exception {

        float h = 1.0f, sigma = 10.0f;
        
        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}
        
        
        for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
            // for each doc in RF-set... m = 1, ..., M
            int luceneDocId = docEntrySet.getKey();
            DocumentVector docV = docEntrySet.getValue();
            //float docScore = docEntrySet.getValue().getDocScore(); // score of D_m
            for (Map.Entry<String, PerTermStat> entrySet : docV.docPerTermStat.entrySet()) {
                // for each term of that feedback document... w \in D_m
                String key = entrySet.getKey();
                PerTermStat value = entrySet.getValue();
                
                //float P_wGivenM = get_P_wGivenM(key); // P(w|M)

                String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
                String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");
//                String[] composedTerms = query.composition.split("\\s+");
//                String[] composedQuery = getUniqTerms((String[]) ArrayUtils.addAll(analyzedQuery, composedTerms)).toArray(new String[0]);

                float KDERLMScore = 0.0f;
                for (String qTerm : analyzedQuery) { // without word composition
                //for (String qTerm : composedQuery) { // with word composition
                    // for each query term... q_i \in Q
                    
                    //float P_qiGivenM = get_P_wGivenM(qTerm); // P(q_i|M)
                    for (Map.Entry<Integer, DocumentVector> docEntrySet1 : feedbackDocumentVectors.entrySet()) {
                    // for each doc in RF-set... j = 1, ..., M
                        int luceneDocId1 = docEntrySet1.getKey();
                        DocumentVector docV1 = docEntrySet1.getValue();
                        float docScore = docEntrySet1.getValue().getDocScore(); // score of D_j

                        //float alpha = P_wGivenM * P_qiGivenM;
                        float alpha = return_Smoothed_MLE_Log(key, feedbackDocumentVectors.get(luceneDocId1)) * // P(w|D_j)
                                        return_Smoothed_MLE_Log(qTerm, feedbackDocumentVectors.get(luceneDocId1)); // P(q_i|D_j)
                        //float dist1 = getEuclideanDistance(key, qTerm);
                        //float dist1 = getEuclideanDistance_BERT(key, qTerm, query);
                        float dist1 = getEuclideanDistance_W2V_and_BERT(key, qTerm, query);
//                        float dist2 = return_Smoothed_MLE_Log(key, feedbackDocumentVectors.get(luceneDocId)) - // P(w|D_m)
//                                        return_Smoothed_MLE_Log(qTerm, feedbackDocumentVectors.get(luceneDocId1)); // P(q_i|D_j)
                        
                        //KDERLMScore += alpha * gaussianKernel((dist1+dist2)/h, sigma);
                        //KDERLMScore += alpha * gaussianKernel((float)(Math.pow(dist1, 2)+Math.pow(dist2, 2))/h, sigma);
                        KDERLMScore += alpha * gaussianKernel((float)(dist1)/h, sigma) * docScore;
                    }  
                }
                list_PwGivenR.add(new WordProbability(key, KDERLMScore)); 
            }
        }
        
        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    public HashMap KDERM1General(TRECQuery query, TopDocs topDocs) throws Exception {

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            
            String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
            String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");
            
            float P_wGivenM = 0.0f, KDERLMScore = 0.0f;
            //float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;

            for (String qTerm : analyzedQuery) {

                float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;
                for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
                // for each doc in RF-set
                    int luceneDocId = docEntrySet.getKey();
                    DocumentVector docV = docEntrySet.getValue();
                    
                    P_wGivenM += return_Smoothed_MLE_Log(t, feedbackDocumentVectors.get(luceneDocId)); // P(w|D)
                    P_qiGivenM += return_Smoothed_MLE_Log(qTerm, feedbackDocumentVectors.get(luceneDocId)); // P(q_i|D)
                    //P_qiGivenM += return_Smoothed_MLE(qTerm, docV) * hash_P_Q_Given_D.get(luceneDocId); // P(q_i|D) * P(Q|D)
                    P_QGivenM += hash_P_Q_Given_D.get(luceneDocId);
                }
                
                KDERLMScore += P_wGivenM * // P(w|M)
                                P_qiGivenM * // P(q_i|M)
                                //P_QGivenM * // P(Q|M)
                                getKDEScoreForSingleQTermGeneral(t, qTerm, 1.0f); // KDE score of 't' based on 'qTerm' (W2V)
                                //getKDEScoreForSingleQTermGeneral_BERT(t, qTerm, 1.0f, query); // KDE score of 't' based on 'qTerm' (BERT)
            }
            
            list_PwGivenR.add(new WordProbability(t, KDERLMScore));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    public HashMap KDERM1_2(TRECQuery query, TopDocs topDocs) throws Exception {

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            
            String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
            String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");
            
            float P_wGivenM = 0.0f, KDERLMScore = 0.0f;
            //float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;

            for (String qTerm : analyzedQuery) {
                
                float contextApproVal = getTermLevelContextualAppropriateness(query, t);
                //float contextApproVal = getTermLevelContextualAppropriateness_singleContextBased(query, t);

                float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;
                for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
                // for each doc in RF-set
                    int luceneDocId = docEntrySet.getKey();
                    DocumentVector docV = docEntrySet.getValue();
                    
                    P_wGivenM += return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)); // P(w|D)
                    P_qiGivenM += return_Smoothed_MLE(qTerm, docV) * hash_P_Q_Given_D.get(luceneDocId); // P(q_i|D) * P(Q|D)
                    P_QGivenM += hash_P_Q_Given_D.get(luceneDocId);
                }
                
                KDERLMScore += P_wGivenM * // P(w|M)
                                P_qiGivenM * // P(q_i|M)
                                //P_QGivenM * // P(Q|M)
                                contextApproVal *
                                getKDEScoreForSingleQTerm(t, qTerm, 1.0f); // KDE score of 't' based on 'qTerm'
            }
            
            list_PwGivenR.add(new WordProbability(t, KDERLMScore));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    public HashMap KDERM1_2General(TRECQuery query, TopDocs topDocs) throws Exception {

        list_PwGivenR = new ArrayList<>();

        hashmap_PwGivenR = new LinkedHashMap<>();

        // Calculating for each wi in R: P(wi|R)~P(wi, q1 ... qk)
        // P(wi, q1 ... qk) = \sum{d\in PRD} {P(w|D)*\prod_{i=1... k} {P(qi|D}}

        for (Map.Entry<String, PerTermStat> entrySet : feedbackTermStats.entrySet()) {
            // for each t in R:
            String t = entrySet.getKey();
            
            String fieldToSearch = this.fieldForFeedback; // Here fieldToSearch = fieldForFeedback
            String[] analyzedQuery = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");
            
            float P_wGivenM = 0.0f, KDERLMScore = 0.0f;
            //float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;

            for (String qTerm : analyzedQuery) {

                float P_qiGivenM = 0.0f, P_QGivenM = 0.0f;
                for (Map.Entry<Integer, DocumentVector> docEntrySet : feedbackDocumentVectors.entrySet()) {
                // for each doc in RF-set
                    int luceneDocId = docEntrySet.getKey();
                    DocumentVector docV = docEntrySet.getValue();
                    
                    P_wGivenM += return_Smoothed_MLE(t, feedbackDocumentVectors.get(luceneDocId)); // P(w|D)
                    P_qiGivenM += return_Smoothed_MLE(qTerm, docV) * hash_P_Q_Given_D.get(luceneDocId); // P(q_i|D) * P(Q|D)
                    P_QGivenM += hash_P_Q_Given_D.get(luceneDocId);
                }
                
                KDERLMScore += P_wGivenM * // P(w|M)
                                P_qiGivenM * // P(q_i|M)
                                //P_QGivenM * // P(Q|M)
                                getKDEScoreForSingleQTermGeneral(t, qTerm, 1.0f); // KDE score of 't' based on 'qTerm'
            }
            
            list_PwGivenR.add(new WordProbability(t, KDERLMScore));
        }

        // ++ sorting list in descending order
        Collections.sort(list_PwGivenR, new Comparator<WordProbability>(){
            @Override
            public int compare(WordProbability t, WordProbability t1) {
                return t.p_w_given_R<t1.p_w_given_R?1:t.p_w_given_R==t1.p_w_given_R?0:-1;
            }});
        // -- sorted list in descending order

        for (WordProbability singleTerm : list_PwGivenR) {
            if (null == hashmap_PwGivenR.get(singleTerm.w)) {
                hashmap_PwGivenR.put(singleTerm.w, new WordProbability(singleTerm.w, singleTerm.p_w_given_R));
            }
            //* else: The t is already entered in the hash-map 
        }

        return hashmap_PwGivenR;
    }
    
    // Returns KDE score of term 'term' based on the query term 'qterm'.
    public float getKDEScoreForSingleQTerm(String term, String qterm, float sigma) throws Exception {

        int term_index, qterm_index, n;
        float h;

        List<Word2vec> observedTerms = new ArrayList<>();
        
        //qterm_index = getTermIndex(qterm);
        qterm_index = getW2VTermIndex(qterm);
        if(qterm_index >= 0) {
            Word2vec qVec = new Word2vec();
            qVec.term = qterm;
            //qVec.vector = convertVectorStringToFloat(W2Vmodel.get(qterm_index));
            qVec.vector = W2V.get(qterm_index).vector;
            observedTerms.add(qVec);
            
            float[] weightArray = new float[1]; // weight array for KDE. Weights for observed terms
            weightArray[0] = 1.0f;
            
            //term_index = getTermIndex(term);
            term_index = getW2VTermIndex(term);
            if(term_index < 0) {
                //System.out.println("||||||||||||||||||||||||||||||||||||||||| CASE!\t" + term);
                //return 0.0f;
                //return 0.000001f;
                //return 0.5f;
                //return 1.0f;
                return gaussianKernel(100, sigma);
            }
            Word2vec wVec = new Word2vec();
            wVec.term = term;
            //wVec.vector = convertVectorStringToFloat(W2Vmodel.get(term_index));
            wVec.vector = W2V.get(term_index).vector;
            
            n = 1;
            h = (float) Math.pow(((4*Math.pow(sigma, 5))/(3*n)), -(1/5));

            //return getKDE(wVec.vector, observedTerms, weightArray, 1, h, sigma);
            return getKDEDirect(wVec.vector, qVec.vector, h, sigma);
        }

        //return 0.0f;
        //return 0.000001f;
        //return 0.5f;
        //return 1.0f;
        return gaussianKernel(100, sigma);
    }
    
    // Returns KDE score of term 'term' based on the query term 'qterm'.
    public float getKDEScoreForSingleQTermGeneral(String term, String qterm, float sigma) throws Exception {

        int term_index, qterm_index, n;
        float h;

        List<Word2vec> observedTerms = new ArrayList<>();
        
        qterm_index = getW2VTermIndex(qterm);
        if(qterm_index >= 0) {
            Word2vec qVec = new Word2vec();
            qVec.term = qterm;
            qVec.vector = W2V.get(qterm_index).vector;
            observedTerms.add(qVec);
            
            float[] weightArray = new float[1]; // weight array for KDE. Weights for observed terms
            weightArray[0] = 1.0f;
            
            term_index = getW2VTermIndex(term);
            if(term_index < 0) {
                //System.out.println("||||||||||||||||||||||||||||||||||||||||| CASE!\t" + term);
                //return 0.0f;
                //return 0.000001f;
                //return 0.5f;
                //return 1.0f;
                return gaussianKernel(100, sigma);
            }
            Word2vec wVec = new Word2vec();
            wVec.term = term;
            wVec.vector = W2V.get(term_index).vector;
            
            n = 1;
            //h = (float) Math.pow(((4*Math.pow(sigma, 5))/(3*n)), -(1/5));
            h = 1;

            //return getKDE(wVec.vector, observedTerms, weightArray, 1, h, sigma);
            return getKDEDirect(wVec.vector, qVec.vector, h, sigma);
        }

        //return 0.0f;
        //return 0.000001f;
        //return 0.5f;
        //return 1.0f;
        return gaussianKernel(100, sigma);
    }
    
    // Returns KDE score of term 'term' based on the query term 'qterm' (BERT context embedding).
    public float getKDEScoreForSingleQTermGeneral_BERT(String term, String qterm, float sigma, TRECQuery query) throws Exception {

        int term_index, qterm_index, n;
        float h;

        //List<Word2vec> observedTerms = new ArrayList<>();
        
        //qterm_index = getW2VTermIndex(qterm);
        float[] qVec = getBERTvec(qterm, query.qid);
        //if(qterm_index >= 0) {
        if(qVec[0] != -999.0f) {
            //Word2vec qVec = new Word2vec();
            //qVec.term = qterm;
            //qVec.vector = W2V.get(qterm_index).vector;
            //observedTerms.add(qVec);
            
            float[] weightArray = new float[1]; // weight array for KDE. Weights for observed terms
            weightArray[0] = 1.0f;
            
            //term_index = getW2VTermIndex(term);
            float[] wVec = getBERTvec(term, query.qid);
            //if(term_index < 0) {
            if(wVec[0] == -999.0f) {
                //System.out.println("||||||||||||||||||||||||||||||||||||||||| CASE!\t" + term);
                //return 0.0f;
                //return 0.000001f;
                //return 0.5f;
                //return 1.0f;
                return gaussianKernel(100, sigma);
            }
            //Word2vec wVec = new Word2vec();
            //wVec.term = term;
            //wVec.vector = W2V.get(term_index).vector;
            
            n = 1;
            h = (float) Math.pow(((4*Math.pow(sigma, 5))/(3*n)), -(1/5));
            //h = 1;

            //return getKDE(wVec.vector, observedTerms, weightArray, 1, h, sigma);
            return getKDEDirect(wVec, qVec, h, sigma);
        }

        //return 0.0f;
        //return 0.000001f;
        //return 0.5f;
        //return 1.0f;
        return gaussianKernel(100, sigma);
    }
    
    public float getKDEDirect(float[] w, float[] q_i, float h, float sigma) throws Exception {
        float dist, score;
        
        dist = euclideanDistance(normalizeVec(w), normalizeVec(q_i));
        score = gaussianKernel(dist/h, sigma);

        return score;
    }
    
    // Returns the contextual appropriateness (fine-grained) score (max) of term 'term' for the query 'query' (~joint context). For experiments on general data (TREC8).
    public float getTermLevelContextualAppropriatenessGeneral (TRECQuery query, String term) throws Exception {

        String[] terms = query.luceneQuery.toString(fieldForFeedback).split(" ");
        
        float sim = getCosineSimilarityMultiTermsGeneral(terms[0], term);
        float sum = sim;
        float min = sim;
        float max = sim;
        for (int j = 1; j < terms.length; ++j) {
                //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarity(tags[i], terms[j]) + "\tNormalized: " + getCosineSimilarityNormalized(tags[i], terms[j]));
            //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarityMultiTerms2(tags[i], terms[j]) + "\t");
            //sim = getCosineSimilarity(tags[i], terms[j]);
            sim = getCosineSimilarityMultiTermsGeneral(terms[j], term);
            if (sim < min) {
                min = sim;
            }
            if (sim > max) {
                max = sim;
            }
            sum += sim;
        }
        //System.out.println("-------------------------------------");
        float avg = sum / terms.length;
        
        if(max < 0.0f)
            return 0.0f;
        else
            return max;
    }

    // Returns the contextual appropriateness (fine-grained) score (max) of term 'term' for the query 'query' (joint context)
    public float getTermLevelContextualAppropriateness (TRECQuery query, String term) throws Exception {
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        String jointContext = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        //String[] terms = getContextualApproTerms(jointContext);
        String[] terms = getParsedTerms(getContextualApproTerms(jointContext));
        
        float sim = getCosineSimilarityMultiTerms(terms[0], term);
        float sum = sim;
        float min = sim;
        float max = sim;
        for (int j = 1; j < terms.length; ++j) {
                //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarity(tags[i], terms[j]) + "\tNormalized: " + getCosineSimilarityNormalized(tags[i], terms[j]));
            //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarityMultiTerms2(tags[i], terms[j]) + "\t");
            //sim = getCosineSimilarity(tags[i], terms[j]);
            sim = getCosineSimilarityMultiTerms(terms[j], term);
            if (sim < min) {
                min = sim;
            }
            if (sim > max) {
                max = sim;
            }
            sum += sim;
        }
        //System.out.println("-------------------------------------");
        float avg = sum / terms.length;
        
        if(max < 0.0f)
            return 0.0f;
        else
            return max;
    }
    
    // Returns the contextual appropriateness (fine-grained) score (max) of term 'term' for the query 'query' (joint context)
    public float getTermLevelContextualAppropriateness_singleContextBased (TRECQuery query, String term) throws Exception {
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        String grouptypeContext = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-");
        String triptypeContext = "Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-");
        String tripdurationContext = "Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        
        float grouptypeScore = getContextApproScoreForSingleContext(grouptypeContext, term);
        float triptypeScore = getContextApproScoreForSingleContext(triptypeContext, term);
        float tripdurationScore = getContextApproScoreForSingleContext(tripdurationContext, term);

        return (grouptypeScore + triptypeScore + tripdurationScore) / 3.0f;
    }
    
    public float getContextApproScoreForSingleContext(String context, String term)  throws Exception {
        List<TermList> terms = getContextualApproTermsWeighted(context);
        int n = terms.size();
        for (int i = 0; i < n; ++i) {
            if(term.equals(terms.get(i).term)) {
                //return (float) terms.get(i).weight;   // Mohammad's assessor's score [-1.0, +1.0]
                return normalizeMinMax((float) terms.get(i).weight, 1.0f, -1.0f);   // Normalized [0, +1.0]
            }
        }
        return 0.5f;
    }
    
    public float normalizeMinMax(float value, float max, float min) throws Exception {
        return (value - min) / (max - min);
    }
    
    // Returns 'hits' with scores normalized between 0 and 1
    public ScoreDoc[] normalizeMinMax_hits(ScoreDoc[] hits) throws Exception {
        float max = hits[0].score;
        float min = hits[0].score;
        for (int k = 1; k < hits.length; ++k) {
            if (hits[k].score > max) {
                max = hits[k].score;
            }
            if (hits[k].score < min) {
                min = hits[k].score;
            }
        }
        for (int k = 0; k < hits.length; ++k) {
            hits[k].score = (hits[k].score - min) / (max - min);
        }

        return hits;
    }
    
    // Returns 'hits' with scores normalized between 0 and 1, equidistant
    public ScoreDoc[] normalizeEquiDist_hits(ScoreDoc[] hits) throws Exception {
        float score = 1.0f, deduction = 1.0f/(hits.length-1);
        for (int k = 0; k < hits.length; ++k) {
            hits[k].score = score - (k * deduction);
        }

        return hits;
    }

    public int getTreccsQueryJsonIndex(String qID)  throws Exception {    // Get index of query of 'TRECId' in 'treccsQueryJson'
        TRECCSQuery temp = new TRECCSQuery();
        temp.qID = qID;
        return Collections.binarySearch(treccsQueryJson, temp, new cmpTRECCSQuery());
    }

    public class cmpTRECCSQuery implements Comparator<TRECCSQuery> {
        @Override
        public int compare (TRECCSQuery a, TRECCSQuery b) {
            return a.qID.compareTo(b.qID)>0?1:a.qID.compareTo(b.qID)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    // Get contextual appropriate terms weighted (i.e. contextualApproTerms.get(i).posTagsWeighted) for single/joint context 'jointContext'
    public List<TermList> getContextualApproTermsWeighted (String context) throws Exception {
        int n = contextualApproTerms.size();
        for (int i = 0; i < n; ++i) {
            if(context.equals(contextualApproTerms.get(i).context)) {
                return contextualApproTerms.get(i).posTagsWeighted;
            }
        }
        return null;
    }

    // Get contextual appropriate terms for the joint context 'jointContext'
    public String[] getContextualApproTerms (String jointContext) throws Exception {
        int n = contextualApproTerms.size();
        for (int i = 0; i < n; ++i) {
            if(jointContext.equals(contextualApproTerms.get(i).context)) {
                return contextualApproTerms.get(i).posTags;
            }
        }
        return null;
    }

    // Return parsed term (joined with '-' e.g. amateur-archaeology -> amateur-archaeolog)
    public String parsedTerm (String term) throws Exception {
        TRECQuery query = new TRECQuery();
        query.qid = "700";
        query.fieldToSearch = "";
        query.qtitle = term;
        trecQueryparser.getAnalyzedQuery(query, 1);
        String fieldToSearch = this.fieldForFeedback;
        return query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").replace(" ", "-").replace("--", "-");
    }
    
    public String[] getParsedTerms (String[] terms) throws Exception {
        String[] termsParsed = new String[terms.length];
        for (int i = 0; i < terms.length; ++i) {
            termsParsed[i] = parsedTerm(terms[i]);
        }
        return termsParsed;
    }
    
    public float[] vectorAddition (float[] vec1, float[] vec2) throws Exception {
        for (int i = 0; i < vec1.length; ++i) {
            vec1[i] += vec2[i];
        }
        
        return vec1;
    }
    
    // Returns cosine similarity between two terms 'term1' (multiTerm e.g. American-Restaurant) and 'term2', based on W2V vectors
    public float getCosineSimilarityMultiTerms (String term1, String term2) throws Exception {
        String[] multiTerms = term1.split("-");
        //int index2 = getTermIndex(term2);
        int index2 = getW2VTermIndex(term2);
        
        if(index2 >= 0) {
            int multiTermsFlag = 0;
            float[] vec1 = new float[200];
            float[] vec2 = new float[200];
//            for (int j = 1; j < W2Vmodel.get(index2).length; ++j) {
//                vec2[j-1] = Float.parseFloat(W2Vmodel.get(index2)[j]);
//            }
            vec2 = W2V.get(index2).vector;
            for (int i = 0; i < multiTerms.length; ++i) {
                //int index1 = getTermIndex(multiTerms[i]);
                int index1 = getW2VTermIndex(multiTerms[i]);
                if(index1 >= 0) {
                    multiTermsFlag++;
//                    for (int j = 1; j < W2Vmodel.get(index1).length; ++j) {
//                        vec1[j-1] += Float.parseFloat(W2Vmodel.get(index1)[j]); // Vector addition
//                    }
                    for (int j = 0; j < W2V.get(index1).vector.length; ++j) {
                        vec1[j] += W2V.get(index1).vector[j];
                    }
//                    vec1 = vectorAddition(vec1.clone(), W2V.get(index1).vector.clone()); // vector addition
                }
            }
            if(multiTermsFlag > 0) {
                return cosineSimilarity(vec1, vec2);
            }
        }
        
        return 0.0f;
    }

    // Returns cosine similarity between two terms 'term1' (multiTerm e.g. American-Restaurant) and 'term2', based on W2V vectors
    public float getCosineSimilarityMultiTermsGeneral (String term1, String term2) throws Exception {
        String[] multiTerms = term1.split("-");
        int index2 = getW2VTermIndex(term2);
        
        if(index2 >= 0) {
            int multiTermsFlag = 0;
            float[] vec1 = new float[200];
            float[] vec2 = W2V.get(index2).vector;

            for (int i = 0; i < multiTerms.length; ++i) {
                int index1 = getW2VTermIndex(multiTerms[i]);
                if(index1 >= 0) {
                    multiTermsFlag++;
                    for (int j = 0; j < W2V.get(index1).vector.length; ++j) {
                        vec1[j] += W2V.get(index1).vector[j];
                    }
                }
            }
            if(multiTermsFlag > 0) {
                return cosineSimilarity(vec1, vec2);
            }
        }
        
        return 0.0f;
    }
    
}
