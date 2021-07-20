/**
 * TODO: queryField[] is not used. 
 * It works on only the title of the query.
 */
package searcher;

import RelevanceFeedback.NewScore;
import RelevanceFeedback.RLM;
import RelevanceFeedback.RelevanceBasedLanguageModel;
import RelevanceFeedback.WordProbability;
import static common.CommonVariables.FIELD_FULL_BOW;
import static common.CommonVariables.FIELD_ID;
import common.DocumentVector;
import common.TRECQuery;
import common.TRECQueryParser;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BlendedTermQuery.Builder;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.similarities.AfterEffect;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.AfterEffectL;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModel;
import org.apache.lucene.search.similarities.BasicModelBE;
import org.apache.lucene.search.similarities.BasicModelD;
import org.apache.lucene.search.similarities.BasicModelG;
import org.apache.lucene.search.similarities.BasicModelIF;
import org.apache.lucene.search.similarities.BasicModelIn;
import org.apache.lucene.search.similarities.BasicModelIne;
import org.apache.lucene.search.similarities.BasicModelP;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.MultiSimilarity;
import org.apache.lucene.search.similarities.Normalization;
import org.apache.lucene.search.similarities.Normalization.NoNormalization;
import org.apache.lucene.search.similarities.NormalizationH1;
import org.apache.lucene.search.similarities.NormalizationH2;
import org.apache.lucene.search.similarities.NormalizationH3;
import org.apache.lucene.search.similarities.NormalizationZ;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import NaiveBayse.de.daslaboratorium.machinelearning.classifier.bayes.BayesClassifier;
import NaiveBayse.de.daslaboratorium.machinelearning.classifier.Classifier;
import java.util.Iterator;
import java.util.Random;
import org.apache.commons.lang.ArrayUtils;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.MultiFields;
import org.json.JSONArray;
import org.json.JSONObject;

public class WebDocSearcher_TRECCS_Novel {

    String          propPath;
    Properties      prop;
    IndexReader     indexReader;
    IndexSearcher   indexSearcher;
    String          indexPath;
    File            indexFile;
    String          stopFilePath;
    String          queryPath;
    File            queryFile;      // the query file
    int             queryFieldFlag; // 1. title; 2. +desc, 3. +narr
    String          []queryFields;  // to contain the fields of the query to be used for search
    Analyzer        analyzer;
    Analyzer        webDocAnalyzer;           // webDocAnalyzer

    String          runName;
    int             numHits;
    boolean         boolIndexExists;
    String          resPath;        // path of the res file
    FileWriter      resFileWriter;  // the res file writer
    List<TRECQuery> queries;
    TRECQueryParser trecQueryparser;
    String          fieldToSearch;
    int             simFuncChoice;
    float           param1, param2, param3, param4;
    
    String          TRECCSTagsClustersFilePath; // Path of the (manual) clusters (of TREC CS tags) file
    String          TRECCSTagsClustersWeightFilePath;
    String          W2VFilePath; // Path of the Word2vec trained .vec file
    String          BERTFilePath;
    String          BERTDocFilePath;
    String          queryWiseUserPrefFilePath;
    String          queryWiseUserPrefTagsFilePath;
    String          queryWiseUserPrefNegativeTagsFilePath;
    String          foursquareDataFilePath;
    String          treccsQueryJsonFilePath;
    List<String []> TagsClusters; // To store TREC CS tags' clusters
    List<String []> TagsClustersWeight;
    List<String []> W2Vmodel; // To store all the terms and their (200D) vectors, generated by Word2vec
    List<Word2vec>  W2V;
    List<BERT>      bert;
    List<BERTDoc>   BERTdoc;
    //List<TRECQuery> subQueries; // To store sub-queries (broken as per TREC CS tags clusters) for each query
    
//    List<String []> queryWiseUserPref;
//    List<int []>   queryWiseUserPrefRating;
    List<UserPreference>    userPref;
    List<UserPreference>    userPrefTags;
    List<UserPreference>    userPrefNegativeTags;
    List<UserPreference>    qrels;
    List<FourSquareData>    foursquareData;
    List<TRECCSQuery>       treccsQueryJson;
    String[]                context;
    String[]                category;
    float[]                 contextCategoryScore;
    //float[][]               ContextVsCategory;
    
    List<ContextualQuery>               contextualQuery;
    List<ContextualQuery>               contextualQueryManual;
    List<ContextualQuery>               contextualApproTerms;
    List<ContextualAppropriateness>     contextualAppropriateness;
    List<PoiContextualAppropriateness>  poiWiseContextualAppropriateness;
    
    RLM     rlm;    // Relevance model
    int     numFeedbackDocsGlobal;
    int     numFeedbackTermsGlobal;
    float   QMIXGlobal;
    
    long collectionSizeGlobal;
    long nDocGlobal;      // total number of documents in the index
    
    List<DocumentSimilarity>    docSim;

    public WebDocSearcher_TRECCS_Novel(String propPath) throws IOException, Exception {

        this.propPath = propPath;
        prop = new Properties();
        try {
            prop.load(new FileReader(propPath));
        } catch (IOException ex) {
            System.err.println("Error: Properties file missing in "+propPath);
            System.exit(1);
        }
        //----- Properties file loaded

        // +++++ setting the analyzer with English Analyzer with Smart stopword list
        stopFilePath = prop.getProperty("stopFilePath");
        System.out.println("stopFilePath set to: " + stopFilePath);
        common.EnglishAnalyzerWithSmartStopword engAnalyzer = new common.EnglishAnalyzerWithSmartStopword(stopFilePath);
        analyzer = engAnalyzer.setAndGetEnglishAnalyzerWithSmartStopword();
        // ----- analyzer set: analyzer
        webDocAnalyzer = new common.WebDocAnalyzer();

        //+++++ index path setting 
        indexPath = prop.getProperty("indexPath");
        System.out.println("indexPath set to: " + indexPath);
        indexFile = new File(indexPath);
        Directory indexDir = FSDirectory.open(indexFile.toPath());

        if (!DirectoryReader.indexExists(indexDir)) {
            System.err.println("Index doesn't exists in "+indexPath);
            boolIndexExists = false;
            System.exit(1);
        }
        //----- index path set

        /* setting query path */
        queryPath = prop.getProperty("queryPath");
        System.out.println("queryPath set to: " + queryPath);
        queryFile = new File(queryPath);
        queryFieldFlag = Integer.parseInt(prop.getProperty("queryFieldFlag"));
        queryFields = new String[queryFieldFlag-1];
        /* query path set */
        // TODO: queryFields unused

        /* constructing the query */
        fieldToSearch = prop.getProperty("fieldToSearch", FIELD_FULL_BOW);
        System.out.println("Searching field for retrieval: " + fieldToSearch);
        trecQueryparser = new TRECQueryParser(queryPath, analyzer, fieldToSearch);
        queries = constructQueries();
        /* constructed the query */

        simFuncChoice = Integer.parseInt(prop.getProperty("similarityFunction"));
        if (null != prop.getProperty("param1"))
            param1 = Float.parseFloat(prop.getProperty("param1"));
        if (null != prop.getProperty("param2"))
            param2 = Float.parseFloat(prop.getProperty("param2"));
        if (null != prop.getProperty("param3"))
            param3 = Float.parseFloat(prop.getProperty("param3"));
        if (null != prop.getProperty("param4"))
            param4 = Float.parseFloat(prop.getProperty("param4"));

        /* setting indexReader and indexSearcher */
        indexReader = DirectoryReader.open(FSDirectory.open(indexFile.toPath()));

        indexSearcher = new IndexSearcher(indexReader);
        setSimilarityFunction(simFuncChoice, param1, param2, param3, param4);

        setRunName_ResFileName();

        File fl = new File(resPath);
        //if file exists, delete it
        if(fl.exists())
            System.out.println(fl.delete());

        resFileWriter = new FileWriter(resPath, true);

        /* res path set */
        numHits = Integer.parseInt(prop.getProperty("numHits", "1000"));  

        // Ani: Setting the file paths for Word2vec model and TREC CS tags' cluster
        TRECCSTagsClustersFilePath = prop.getProperty("TRECCSTagsClustersFilePath");
        TRECCSTagsClustersWeightFilePath = prop.getProperty("TRECCSTagsClustersWeightFilePath");
        W2VFilePath = prop.getProperty("W2VFilePath");
        BERTFilePath = prop.getProperty("BERTFilePath");
        BERTDocFilePath = prop.getProperty("BERTDocFilePath");
        queryWiseUserPrefFilePath = prop.getProperty("queryWiseUserPrefFilePath");
        queryWiseUserPrefTagsFilePath = prop.getProperty("queryWiseUserPrefTagsFilePath");
        queryWiseUserPrefNegativeTagsFilePath = prop.getProperty("queryWiseUserPrefNegativeTagsFilePath");
        foursquareDataFilePath = prop.getProperty("foursquareDataFilePath");
        treccsQueryJsonFilePath = prop.getProperty("treccsQueryJsonFilePath");
        System.out.println("W2VFilePath set to: " + W2VFilePath + "\nBERTFilePath set to: " + BERTFilePath + "\nBERTDocFilePath set to: " + BERTDocFilePath + "\nTRECCSTagsClustersFilePath set to: " + TRECCSTagsClustersFilePath + "\nTRECCSTagsClustersWeightFilePath set to: " + TRECCSTagsClustersWeightFilePath + "\nqueryWiseUserPrefFilePath set to: " + queryWiseUserPrefFilePath + "\nqueryWiseUserPrefTagsFilePath set to: " + queryWiseUserPrefTagsFilePath + "\nqueryWiseUserPrefNegativeTagsFilePath set to: " + queryWiseUserPrefNegativeTagsFilePath + "\nfoursquareDataFilePath set to: " + foursquareDataFilePath + "\ntreccsQueryJsonFilePath set to: " + treccsQueryJsonFilePath);
    }

    private void setSimilarityFunction(int choice, float param1, float param2, float param3, float param4) {

        switch(choice) {
            case 0:
                indexSearcher.setSimilarity(new DefaultSimilarity());
                System.out.println("Similarity function set to DefaultSimilarity");
                break;
            case 1:
                indexSearcher.setSimilarity(new BM25Similarity(param1, param2));
                System.out.println("Similarity function set to BM25Similarity"
                    + " with parameters: " + param1 + " " + param2);
                break;
            case 2:
                indexSearcher.setSimilarity(new LMJelinekMercerSimilarity(param1));
                System.out.println("Similarity function set to LMJelinekMercerSimilarity"
                    + " with parameter: " + param1);
                break;
            case 3:
                indexSearcher.setSimilarity(new LMDirichletSimilarity(param1));
                System.out.println("Similarity function set to LMDirichletSimilarity"
                    + " with parameter: " + param1);
                break;
            case 4:
//                indexSearcher.setSimilarity(new DFRSimilarity(new BasicModelIF(), new AfterEffectB(), new NormalizationH2()));
                BasicModel bm;
                AfterEffect ae;
                Normalization nor;
                switch((int)param1){
                    case 1:
                        bm = new BasicModelBE();
                        break;
                    case 2:
                        bm = new BasicModelD();
                        break;
                    case 3:
                        bm = new BasicModelG();
                        break;
                    case 4:
                        bm = new BasicModelIF();
                        break;
                    case 5:
                        bm = new BasicModelIn();
                        break;
                    case 6:
                        bm = new BasicModelIne();
                        break;
                    case 7:
                        bm = new BasicModelP();
                        break;
                    default:
                        bm = new BasicModelIF();
                        break;
                }
                switch ((int)param2){
                    case 1:
                        ae = new AfterEffectB();
                        break;
                    case 2:
                        ae = new AfterEffectL();
                        break;
                    default:
                        ae = new AfterEffectB();
                        break;
                }
                switch ((int)param3) {
                    case 1:
                        nor = new NormalizationH1();
                        break;
                    case 2:
                        nor = new NormalizationH2();
                        break;
                    case 3:
                        nor = new NormalizationH3();
                        break;
                    case 4:
                        nor = new NormalizationZ();
                        break;
                    case 5:
                        nor = new NoNormalization();
                        break;
                    default:
                        nor = new NormalizationH2();
                        break;
                }
//                bm = new BasicModelIF();
                indexSearcher.setSimilarity(new DFRSimilarity(bm, ae, nor));
                System.out.println("Similarity function set to DFRSimilarity with default parameters");
                break;
            case 5:
                Similarity[] sims = {
                        new BM25Similarity(param1, param2),
                        new LMJelinekMercerSimilarity(param3),
                        new LMDirichletSimilarity(param4),
                        //new DFRSimilarity(new BasicModelBE(), new AfterEffectB(), new NormalizationH1()),
                    };
                Similarity sim = new MultiSimilarity(sims);

                indexSearcher.setSimilarity(sim);
                System.out.println("Similarity function set to CombSUM(BM25, LM-JM, LM-Di)"
                    + " with parameters: k1=" + param1 + ", b=" + param2 + " (BM25); lamda=" + param3 + " (LM-JM); mu=" + param4 + " (LM-Di)");
                break;
        }
    }

    private void setRunName_ResFileName() {

        runName = queryFile.getName()+"-"+fieldToSearch+"-"+indexSearcher.getSimilarity(true).
            toString().replace(" ", "-").replace("(", "").replace(")", "");
        if(null == prop.getProperty("resPath"))
            resPath = "/home/anirban/";
        else
            resPath = prop.getProperty("resPath");
        if(!resPath.endsWith("/"))
            resPath = resPath+"/";
        resPath = resPath+runName;
        System.out.println("Result will be stored in: "+resPath);
    }

    private List<TRECQuery> constructQueries() throws Exception {

        trecQueryparser.queryFileParse();
        return trecQueryparser.queries;
    }

//    class SearchResult {
//        ScoreDoc docScore;
//        Document doc;
//        SearchResult(ScoreDoc docScore, Document doc){
//            this.docScore = docScore;
//            this.doc = doc;
//        }
//    }
//    
//    public class ReRankAni implements Comparator<SearchResult> {
//        @Override
//        public int compare (SearchResult a, SearchResult b) {
//            return a.docScore.score>b.docScore.score?1:a.docScore.score==b.docScore.score?0:-1;
//        }
//    }
    public class cmpScoreDoc implements Comparator<ScoreDoc> {
        @Override
        public int compare (ScoreDoc a, ScoreDoc b) {
            //return a.score>b.score?1:a.score==b.score?0:-1;   // standard sort (ascending order)
            return a.score<b.score?1:a.score==b.score?0:-1; // reverse order
        }
    }

    public class cmpW2VModel implements Comparator<String []> {
        @Override
        public int compare (String a[], String b[]) {
            return a[0].compareTo(b[0])>0?1:a[0].compareTo(b[0])==0?0:-1;   // standard sort (ascending order)
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
    
    public class cmpBERTDoc implements Comparator<BERTDoc> {
        @Override
        public int compare (BERTDoc a, BERTDoc b) {
            return a.docID.compareTo(b.docID)>0?1:a.docID.compareTo(b.docID)==0?0:-1;
            //return a.consineScore>b.consineScore?1:a.consineScore==b.consineScore?0:-1;   // standard sort (ascending order)
            //return a.consineScore<b.consineScore?1:a.consineScore==b.consineScore?0:-1; // reverse order
        }
    }

    public class cmpW2VCosineSim implements Comparator<Word2vec> {
        @Override
        public int compare (Word2vec a, Word2vec b) {
            //return a.consineScore>b.consineScore?1:a.consineScore==b.consineScore?0:-1;   // standard sort (ascending order)
            return a.consineScore<b.consineScore?1:a.consineScore==b.consineScore?0:-1; // reverse order
        }
    }

    public class cmpW2VKDESim implements Comparator<Word2vec> {
        @Override
        public int compare (Word2vec a, Word2vec b) {
            //return a.consineScore>b.consineScore?1:a.consineScore==b.consineScore?0:-1;   // standard sort (ascending order)
            return a.KDEScore<b.KDEScore?1:a.KDEScore==b.KDEScore?0:-1; // reverse order
        }
    }

    public class cmpMultiRankList implements Comparator<MultipleRanklists> {
        @Override
        public int compare (MultipleRanklists a, MultipleRanklists b) {
            //return a.score>b.score?1:a.score==b.score?0:-1;   // standard sort (ascending order)
            return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }

    public class cmpMergedRankList implements Comparator<MergedRanklists> {
        @Override
        public int compare (MergedRanklists a, MergedRanklists b) {
            return a.docId.compareTo(b.docId)>0?1:a.docId.compareTo(b.docId)==0?0:-1;   // standard sort (ascending order)
            //return a.docId<b.docId?1:a.docId==b.docId?0:-1; // reverse order
        }
    }
    
    public class cmpTermListWeightAscending implements Comparator<TermList> {
        @Override
        public int compare (TermList a, TermList b) {
            return a.weight>b.weight?1:a.weight==b.weight?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }

    public class cmpTermListWeight implements Comparator<TermList> {
        @Override
        public int compare (TermList a, TermList b) {
            //return a.score>b.score?1:a.score==b.score?0:-1;   // standard sort (ascending order)
            return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }

    public class cmpTermListTerm implements Comparator<TermList> {
        @Override
        public int compare (TermList a, TermList b) {
            return a.term.compareTo(b.term)>0?1:a.term.compareTo(b.term)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpContextTerm implements Comparator<Context> {
        @Override
        public int compare (Context a, Context b) {
            return a.term.compareTo(b.term)>0?1:a.term.compareTo(b.term)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpContextQID implements Comparator<Context> {
        @Override
        public int compare (Context a, Context b) {
            return a.qID.compareTo(b.qID)>0?1:a.qID.compareTo(b.qID)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpDocVec implements Comparator<DocumentVec> {
        @Override
        public int compare (DocumentVec a, DocumentVec b) {
            return a.docID.compareTo(b.docID)>0?1:a.docID.compareTo(b.docID)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpDocumentCluster implements Comparator<DocumentCluster> {
        @Override
        public int compare (DocumentCluster a, DocumentCluster b) {
            //return a.score>b.score?1:a.score==b.score?0:-1;   // standard sort (ascending order)
            return a.clusterScore<b.clusterScore?1:a.clusterScore==b.clusterScore?0:-1; // reverse order
        }
    }
    
    public class cmpDocumentSimilarity implements Comparator<DocumentSimilarity> {
        @Override
        public int compare (DocumentSimilarity a, DocumentSimilarity b) {
            return a.q_d_d.compareTo(b.q_d_d)>0?1:a.q_d_d.compareTo(b.q_d_d)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpFourSquareData implements Comparator<FourSquareData> {
        @Override
        public int compare (FourSquareData a, FourSquareData b) {
            return a.TRECId.compareTo(b.TRECId)>0?1:a.TRECId.compareTo(b.TRECId)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpPoiContextualAppropriateness implements Comparator<PoiContextualAppropriateness> {
        @Override
        public int compare (PoiContextualAppropriateness a, PoiContextualAppropriateness b) {
            return a.TRECId.compareTo(b.TRECId)>0?1:a.TRECId.compareTo(b.TRECId)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpContextQuery implements Comparator<ContextualQuery> {
        @Override
        public int compare (ContextualQuery a, ContextualQuery b) {
            return a.context.compareTo(b.context)>0?1:a.context.compareTo(b.context)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpTRECCSQuery implements Comparator<TRECCSQuery> {
        @Override
        public int compare (TRECCSQuery a, TRECCSQuery b) {
            return a.qID.compareTo(b.qID)>0?1:a.qID.compareTo(b.qID)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpGeocode implements Comparator<Geocode> {
        @Override
        public int compare (Geocode a, Geocode b) {
            return a.ID.compareTo(b.ID)>0?1:a.ID.compareTo(b.ID)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public static class cmpQrel implements Comparator<Qrel> {
        @Override
        public int compare (Qrel a, Qrel b) {
            return a.qID.compareTo(b.qID)>0?1:a.qID.compareTo(b.qID)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }
    
    public class cmpTRECQuery implements Comparator<TRECQuery> {
        @Override
        public int compare (TRECQuery a, TRECQuery b) {
            return a.qid.compareTo(b.qid)>0?1:a.qid.compareTo(b.qid)==0?0:-1;   // standard sort (ascending order)
            //return a.weight<b.weight?1:a.weight==b.weight?0:-1; // reverse order
        }
    }

    public int docExist2(ScoreDoc[] hits, ScoreDoc doc) throws Exception {
        for (int i = 0; i < hits.length; ++i) {
//            System.out.println("Doc No: " + indexSearcher.doc(doc.doc).get("docid") + "\t" + indexReader.document(doc.doc).get("docid"));
//            System.exit(1);
            //if(indexSearcher.doc(hits[i].doc).get("docid").equals(indexSearcher.doc(doc.doc).get("docid"))) {
            //if(indexSearcher.doc(hits.get(i).doc).get("docid").equals(indexSearcher.doc(doc.doc).get("docid"))) {
            if(indexSearcher.doc(hits[i].doc).get("docid").equals(indexSearcher.doc(doc.doc).get("docid"))) {
                return i;
            }
        }
        return -1;
    }

    public int docExist(List<ScoreDoc> hits, int count, ScoreDoc doc) throws Exception {
        int n = hits.size();
        for (int i = 0; i < n; ++i) {
//            System.out.println("Doc No: " + indexSearcher.doc(doc.doc).get("docid") + "\t" + indexReader.document(doc.doc).get("docid"));
//            System.exit(1);
            //if(indexSearcher.doc(hits[i].doc).get("docid").equals(indexSearcher.doc(doc.doc).get("docid"))) {
            if(indexSearcher.doc(hits.get(i).doc).get("docid").equals(indexSearcher.doc(doc.doc).get("docid"))) {
                return i;
            }
        }
        return -1;
    }

    //public ScoreDoc[] mergeRanklists(List<ScoreDoc []> hitsList, float[] hitsListWeight) throws Exception {
    public ScoreDoc[] mergeRanklists(List<MultipleRanklists> multiRankList) throws Exception {
        //ScoreDoc[] hits = new ScoreDoc[numHits];
        List<ScoreDoc> hits = new ArrayList<>();
        //List<MergedRanklists> mergedLists = updateAvgScoreRanklists(multiRankList);
        List<String> nId = new ArrayList<>();
        int nMultiRankList = multiRankList.size();
        for (int i = 0; i < nMultiRankList; ++i) {
            if(multiRankList.get(i).hits.length >= multiRankList.get(i).nDocs) {
                for (int j = 0; j < multiRankList.get(i).nDocs; ++j) {
                    nId.add(indexSearcher.doc(multiRankList.get(i).hits[j].doc).get("docid"));
                }
            }
        }
        List<String> nIdUniq = getUniqTerms(nId.toArray(new String[0]));
        int possibleToRetrieve = nIdUniq.size();
        System.out.println("\n||||||||||||||||||||||||||| possibleToRetrieve: " + possibleToRetrieve);
//        for (int i = 0; i < multiRankList.size(); ++i) {
//            System.out.print("List: " + i + " (Cl: " + multiRankList.get(i).tagClass + ", " + multiRankList.get(i).nDocs + " of " + multiRankList.get(i).hits.length + ")\t");
//        }
//        System.out.println();
        
        int count = 0, j = 0, k = 0, flag = 0;
        int nHits = Math.min(numHits, possibleToRetrieve);
        while (count < nHits) {
            nMultiRankList = multiRankList.size();
            for (int i = 0; i < nMultiRankList; ++i) {
                //if((k < multiRankList.get(i).nDocs) && (count < numHits) && (!"-1".equals(multiRankList.get(i).tagClass))) {
                if((count < numHits) && (multiRankList.get(i).hits != null) && (k < multiRankList.get(i).nDocs) && (k < multiRankList.get(i).hits.length)) {
                    if(count == 0) {
                        hits.add(multiRankList.get(i).hits[k]);
                        //hits[j] = multiRankList.get(i).hits[k];
                        //System.out.println(j + ": List " + i + " " + k + "-th doc of " + multiRankList.get(i).nDocs + "\tCount: " + count);
                        j++; count++;                        
                    }
                    else if(docExist(hits, count, multiRankList.get(i).hits[k]) < 0) {
                        hits.add(multiRankList.get(i).hits[k]);
                        //hits[j] = multiRankList.get(i).hits[k];
                        //System.out.println(j + ": List " + i + " " + k + "-th doc of " + multiRankList.get(i).nDocs + "\tCount: " + count);
                        j++; count++;                        
                    }
                }
//                else if(multiRankList.get(i).nDocs > multiRankList.get(i).hits.length) {
//                    flag = 1; break;
//                }
            }
//            if(flag == 1) {
//                break;
//            }
            k++;
        }

        //Arrays.sort(hits, new cmpScoreDoc());
        Collections.sort(hits, new cmpScoreDoc());
        
//        System.exit(1);

//        for (int i = 0; i < TagsClustersWeight.size(); ++i) {
//            for (int j = 0; j < TagsClustersWeight.get(i).length; ++j) {
//                System.out.print(TagsClustersWeight.get(i)[j] + " ");
//            }
//            System.out.println();
//        }
//        System.exit(1);
        //return multiRankList.get(0).hits;
        if(hits != null) {
            return hits.toArray(new ScoreDoc[0]);
        }
        return null;
    }
    
    public List<MultipleRanklists> updateAvgScoreRanklists(List<MultipleRanklists> multiRankList) throws Exception {  // Take avg score if a doc is present in multiple ranklists.
        
        List<MergedRanklists> mergedLists = new ArrayList<>();
        List<MergedRanklists> mergedListsAvg = new ArrayList<>();
        
        if(multiRankList.size() > 0) {
            int nMultiRankList = multiRankList.size();
            for (int i = 0; i < nMultiRankList; ++i) {
                for (int j = 0; j < multiRankList.get(i).hits.length; ++j) {
                    MergedRanklists tempLists = new MergedRanklists();
                    tempLists.docId = indexSearcher.doc(multiRankList.get(i).hits[j].doc).get("docid");
                    tempLists.score = multiRankList.get(i).hits[j].score;
                    mergedLists.add(tempLists);
                }
            }
            Collections.sort(mergedLists, new cmpMergedRankList());
            
            String docId = mergedLists.get(0).docId;
            float sum = mergedLists.get(0).score;
            float max = mergedLists.get(0).score;
            int count = 1;
            int nMergedLists = mergedLists.size();
            for (int i = 1; i < nMergedLists; ++i) {
                if(docId.equals(mergedLists.get(i).docId)) {
                    if(mergedLists.get(i).score > max) {
                        max = mergedLists.get(i).score;
                    }
                    sum += mergedLists.get(i).score;
                    count++;
                }
                else {
                    MergedRanklists tempLists = new MergedRanklists();
                    tempLists.docId = docId;
                    //tempLists.score = sum / count;  // Avg score from all ranklists
                    tempLists.score = sum;  // Sum of the scores from all ranklists. It is eventually boosting up a doc if the doc appears in multiple ranklists
                    //tempLists.score = max + ((1.0f - max) * count / multiRankList.size());    // Weighting as per #times the doc appeared on different ranklits. Higher the #times a doc appear in different ranklists, higher the score
                    //tempLists.score = (sum / count) + ((1.0f - (sum / count)) * count / multiRankList.size());
                    mergedListsAvg.add(tempLists);
                    docId = mergedLists.get(i).docId;
                    sum = mergedLists.get(i).score;
                    max = mergedLists.get(i).score;
                    count = 1;
                }
            }
            MergedRanklists tempLists = new MergedRanklists();
            tempLists.docId = docId;
            //tempLists.score = sum / count;
            tempLists.score = sum;
            mergedListsAvg.add(tempLists);

//            for (int i = 0; i < multiRankList.size(); ++i) {
//                for (int j = 0; j < multiRankList.get(i).hits.length; ++j) {
//                    System.out.println(indexSearcher.doc(multiRankList.get(i).hits[j].doc).get("docid") + "\t" + multiRankList.get(i).hits[j].score);
//                }
//                System.out.println("----------------------------");
//            }
//            System.out.println("---------------------------------------------------------------");
//            for (int i = 0; i < mergedListsAvg.size(); ++i) {
//                System.out.println(mergedListsAvg.get(i).docId + "\t" + mergedListsAvg.get(i).score);
//            }
            nMultiRankList = multiRankList.size();
            for (int i = 0; i < nMultiRankList; ++i) {
                for (int j = 0; j < multiRankList.get(i).hits.length; ++j) {
                    tempLists = new MergedRanklists();
                    tempLists.docId = indexSearcher.doc(multiRankList.get(i).hits[j].doc).get("docid");
                    int index = Collections.binarySearch(mergedListsAvg, tempLists, new cmpMergedRankList());
                    if(index >= 0) {
                        multiRankList.get(i).hits[j].score = mergedListsAvg.get(index).score;
                    }
                }
            }
        }

        return multiRankList;
    }
    
    public ScoreDoc[] retrieveMultiQuery1(TRECQuery query) throws Exception {
        ScoreDoc[] hits = null;
        List<TRECQuery> subQueries = generateSubQueries(query); 
        for (int i = 0; i < subQueries.size(); ++i) {
            if(!"-1".equals(subQueries.get(i).qClass)) {
                return retrieve(subQueries.get(i));
            }
        }
        return hits;
    }
    
    public ScoreDoc[] retrieveMultiQuery(TRECQuery query) throws Exception {
//        List<ScoreDoc []> hitsList = new ArrayList<>();
//        float[] hitsListWeight = new float[TagsClusters.size()]; // Actual (used) size of hitsListWeight would be equal to the size of hitsList
        List<MultipleRanklists> multiRankList = new ArrayList<>();
        
        int j = 0, nDoc = 0;
        List<TRECQuery> subQueries = generateSubQueries(query);   // Generating multiple sub-queries
        //System.out.println("PP: " + trecQueryparser.getAnalyzedQuery(subQueries.get(0), 1).toString(fieldToSearch).replace("(", "").replace(")", ""));
        //getExpandedQuery(subQueries.get(0));
        //System.out.println(getExpandedQuery(subQueries.get(0)).qtitle);
        //System.exit(1);
        int nSubQuery = subQueries.size();
        for (int i = 0; i < nSubQuery; ++i) {
            if(!"-1".equals(subQueries.get(i).qClass)) {
//                hitsList.add(retrieve(subQueries.get(i)));
//                hitsListWeight[j] = subQueries.get(i).qClassWeight;
                if(retrieveCustomized(subQueries.get(i), numHits) != null) {
                    multiRankList.add(new MultipleRanklists());
                    //multiRankList.get(j).hits = retrieve(subQueries.get(i));    // Retrieval with subquery terms OLD
                    //multiRankList.get(j).hits = retrieveCustomized(subQueries.get(i), 150);  // Retrieval with subquery terms Corrected (discarding docs with no topic contribution)
                    //multiRankList.get(j).hits = retrieve(getExpandedQuery(subQueries.get(i)));    // Retrieval with expanded (top k W2V) subquery terms
                    //multiRankList.get(j).hits = retrieveCustomized(getExpandedQuery(subQueries.get(i), 5), 150);    // Retrieval with expanded (top k W2V) subquery terms
                    //multiRankList.get(j).hits = retrieveCustomized(getExpandedQueryW2VKDE(subQueries.get(i), 5, 100), 150);    // Retrieval with expanded (top k W2V) subquery terms
                    //multiRankList.get(j).hits = retrieveCustomized(getExpandedQueryWithTopTerms(subQueries.get(i), 0.5f, 5), 150);    // Retrieval with expanded (top k LM term selection) subquery terms
                    
                    multiRankList.get(j).hits = RM3Explore(subQueries.get(i), 2, 0.8f, 0.8f); // FRLM
                    
                    
                    multiRankList.get(j).weight = subQueries.get(i).qClassWeight;
                    multiRankList.get(j).nDocs = (int) Math.ceil(subQueries.get(i).qClassWeight * numHits);
                    multiRankList.get(j).tagClass = subQueries.get(i).qClass;
                    nDoc += multiRankList.get(j).nDocs;
                    //j++;
//    //                System.out.println("|||||||||||||||||||| Hits: " + multiRankList.get(j).hits.length);
//                    // Normalizing the scores between 0 and 1
//                    float max = multiRankList.get(j).hits[0].score;
//                    float min = multiRankList.get(j).hits[0].score;
//                    for (int k = 1; k < multiRankList.get(j).hits.length; ++k) {
//                        if(multiRankList.get(j).hits[k].score > max) {
//                            max = multiRankList.get(j).hits[k].score;
//                        }
//                        if(multiRankList.get(j).hits[k].score < min) {
//                            min = multiRankList.get(j).hits[k].score;
//                        }
//                    }
//                    for (int k = 0; k < multiRankList.get(j).hits.length; ++k) {
//                        multiRankList.get(j).hits[k].score = (multiRankList.get(j).hits[k].score - min) / (max - min);
//                    }
                    
                    j++;
                }
                

            }
        }
        Collections.sort(multiRankList, new cmpMultiRankList());
        
        //multiRankList.get(0).nDocs = numHits;   // Set max #docs to consider to cover the shortage from other ranklists
        multiRankList.get(0).nDocs = multiRankList.get(0).hits.length;   // Set max #docs to consider to cover the shortage from other ranklists
//        if(nDoc < numHits) {
//            multiRankList.get(0).nDocs += (numHits - nDoc);
//        }


        //return mergeRanklists(multiRankList); // OK
        return mergeRanklists(updateAvgScoreRanklists(multiRankList)); // OK with avg score
        
        //return multiRankList.get(0).hits;
    }
    
    public ScoreDoc[] retrieveGeneralBooleanQuery(TRECQuery query, BooleanQuery booleanQuery, int nHits) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
        
        ScoreDoc[] hits = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(numHits);
        //Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");

        System.out.println(query.qid+ ": " +booleanQuery.toString(fieldToSearch));

        indexSearcher.search(booleanQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hits = topDocs.scoreDocs;

        //System.out.println("||||||||| MAX: " + max + "\tMIN: " + min + "\tTOTAL: " + hits.length);
        
//        // Normalizing the scores between 0 and 1
//        if(counter > 0) {
//            float max = hits.get(0).score;
//            float min = hits.get(hits.size() - 1).score;
//            for (int k = 0; k < hits.size(); ++k) {
//                if(max - min == 0.0f)
//                    hits.get(k).score = 0.5f; // 0.5f;  // Problem: all docs with same scores
//                else
//                    hits.get(k).score = ((hits.get(k).score - min) / (max - min));
//            }            
//        }

        //System.out.println("\n|||||||||||||||||||||||||||||||||||||\n#HITS: " + hits.length + "\n|||||||||||||||||||||||||||||||||||||");
        
        return hits;
    }
    
    public ScoreDoc[] retrieveCustomizedBooleanQuery(TRECQuery query, BooleanQuery booleanQuery, int nHits) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
        
        List<ScoreDoc> hits = new ArrayList<>();
        //List<ScoreDoc> hits = null;
        ScoreDoc[] hitsTemp = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(indexReader.numDocs());
        //Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");


        System.out.println(query.qid+ ": " +booleanQuery.toString(fieldToSearch));

        indexSearcher.search(booleanQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hitsTemp = topDocs.scoreDocs;
        
        int counter = 0;
        if(hitsTemp != null && hitsTemp.length > 0) {
            for (int i = 0; i < hitsTemp.length && counter < nHits; ++i) {
                if(query.qcity.equals(indexSearcher.doc(hitsTemp[i].doc).get("cityId"))) {
                    ScoreDoc tempScoreDoc = hitsTemp[i];
                    hits.add(tempScoreDoc);
                    counter++;
                }
            }
        }
        //System.out.println("||||||||| MAX: " + max + "\tMIN: " + min + "\tTOTAL: " + hits.length);


//        // Normalizing the scores between 0 and 1
//        if(counter > 0) {
//            float max = hits.get(0).score;
//            float min = hits.get(hits.size() - 1).score;
//            for (int k = 0; k < hits.size(); ++k) {
//                if(max - min == 0.0f)
//                    hits.get(k).score = 0.5f; // 0.5f;  // Problem: all docs with same scores
//                else
//                    hits.get(k).score = ((hits.get(k).score - min) / (max - min));
//            }            
//        }

        //System.out.println("\n|||||||||||||||||||||||||||||||||||||\n#HITS: " + hits.length + "\n|||||||||||||||||||||||||||||||||||||");
        
        if(counter == 0) {
            System.out.println("Nothing found!");
            return null;
        }
        else
            return hits.toArray(new ScoreDoc[0]);
    }
    
    // Retrieve POIs from from candidate cities only (any candidate city in topic_61_Phase1_TRECformat.xml)
    public ScoreDoc[] retrievePOIs_candidateCities(TRECQuery query, int nHits) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
        
        List<ScoreDoc> hits = new ArrayList<>();
        ScoreDoc[] hitsTemp = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(indexReader.numDocs());
        Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");


        System.out.println(query.qid+ ": " +luceneQuery.toString(fieldToSearch));

        indexSearcher.search(luceneQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hitsTemp = topDocs.scoreDocs;
        
        int counter = 0;
        if(hitsTemp != null && hitsTemp.length > 0) {
            for (int i = 0; i < hitsTemp.length && counter < nHits; ++i) {
                String city = indexSearcher.doc(hitsTemp[i].doc).get("cityId");
                if("359".equals(city) || "338".equals(city) || "212".equals(city) || "217".equals(city) || "227".equals(city) || "232".equals(city) || "356".equals(city) || "413".equals(city) || "185".equals(city) || "300".equals(city) || "306".equals(city) || "291".equals(city) || "319".equals(city) || "389".equals(city) || "203".equals(city) || "329".equals(city) || "218".equals(city) || "253".equals(city) || "248".equals(city) || "221".equals(city) || "335".equals(city) || "380".equals(city) || "260".equals(city) || "344".equals(city) || "306".equals(city) || "188".equals(city) || "341".equals(city) || "166".equals(city) || "291".equals(city) || "176".equals(city) || "197".equals(city) || "181".equals(city) || "188".equals(city) || "371".equals(city) || "195".equals(city) || "178".equals(city) || "336".equals(city) || "356".equals(city) || "253".equals(city) || "329".equals(city) || "182".equals(city) || "389".equals(city) || "191".equals(city) || "334".equals(city) || "193".equals(city) || "274".equals(city) || "335".equals(city) || "366".equals(city) || "382".equals(city) || "274".equals(city) || "270".equals(city) || "167".equals(city) || "385".equals(city) || "261".equals(city) || "381".equals(city) || "331".equals(city) || "342".equals(city) || "167".equals(city) || "232".equals(city) || "356".equals(city) || "300".equals(city)) {
                    ScoreDoc tempScoreDoc = hitsTemp[i];
                    hits.add(tempScoreDoc);
                    counter++;
                }
            }
        }
        
        if(counter == 0) {
            System.out.println("Nothing found!");
            return null;
        }
        else
            return hits.toArray(new ScoreDoc[0]);
    }
    
    public ScoreDoc[] retrieveCustomized_Geo_P(TRECQuery query, int nHits) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
        
        List<ScoreDoc> hits = new ArrayList<>();
        //List<ScoreDoc> hits = null;
        ScoreDoc[] hitsTemp = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(indexReader.numDocs());
        Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");


        System.out.println(query.qid+ ": " +luceneQuery.toString(fieldToSearch));

        indexSearcher.search(luceneQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hitsTemp = topDocs.scoreDocs;
        
        // Normalizing the scores between 0 and 1, and taking distance into account
        float max = hitsTemp[0].score;
        float min = hitsTemp[hitsTemp.length - 1].score;
        for (int k = 0; k < hitsTemp.length; ++k) {
            if (max - min == 0.0f) {
                hitsTemp[k].score = 0.5f; // 0.5f;  // Problem: all docs with same scores (OLD. Not really required)
            } else {
                hitsTemp[k].score = ((hitsTemp[k].score - min) / (max - min)); // Normalized
                
                double dist = distance(Double.parseDouble(query.qlat), Double.parseDouble(indexSearcher.doc(hitsTemp[k].doc).get("lat")), Double.parseDouble(query.qlng), Double.parseDouble(indexSearcher.doc(hitsTemp[k].doc).get("lng")), 0.0, 0.0); // distance between user and POI
                double scoreUpdated = hitsTemp[k].score * Math.exp(-Math.pow(dist, 2));
                //double scoreUpdated = (hitsTemp[k].score==0?0.0000000001f * Math.exp(-Math.pow(dist, 2)):hitsTemp[k].score * Math.exp(-Math.pow(dist, 2)));
                hitsTemp[k].score = (float) scoreUpdated;
            }
        }
        Arrays.sort(hitsTemp, new cmpScoreDoc());

        
        int counter = 0;
        if(hitsTemp != null && hitsTemp.length > 0) {
            for (int i = 0; i < hitsTemp.length && counter < nHits; ++i) {
                if(query.qcity.equals(indexSearcher.doc(hitsTemp[i].doc).get("cityId"))) {
                    ScoreDoc tempScoreDoc = hitsTemp[i];
                    hits.add(tempScoreDoc);
                    counter++;
                }
            }
        }
        //System.out.println("||||||||| MAX: " + max + "\tMIN: " + min + "\tTOTAL: " + hits.length);


//        // Normalizing the scores between 0 and 1
//        if(counter > 0) {
//            float max = hits.get(0).score;
//            float min = hits.get(hits.size() - 1).score;
//            for (int k = 0; k < hits.size(); ++k) {
//                if(max - min == 0.0f)
//                    hits.get(k).score = 0.5f; // 0.5f;  // Problem: all docs with same scores
//                else
//                    hits.get(k).score = ((hits.get(k).score - min) / (max - min));
//            }            
//        }

        //System.out.println("\n|||||||||||||||||||||||||||||||||||||\n#HITS: " + hits.length + "\n|||||||||||||||||||||||||||||||||||||");

        
        if(counter == 0) {
            System.out.println("Nothing found!");
            return null;
        }
        else
            return hits.toArray(new ScoreDoc[0]);
    }
    
    public ScoreDoc[] retrieveCustomized_Geo(TRECQuery query, int nHits) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
        
        List<ScoreDoc> hits = new ArrayList<>();
        //List<ScoreDoc> hits = null;
        ScoreDoc[] hitsTemp = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(indexReader.numDocs());
        Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");


        System.out.println(query.qid+ ": " +luceneQuery.toString(fieldToSearch));

        indexSearcher.search(luceneQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hitsTemp = topDocs.scoreDocs;
        
        // Normalizing the scores between 0 and 1, and taking distance into account
        float max = -999; double maxDist = -999;
        float min = 999; double minDist = 999;
        int counter = 0;
        for (int k = 0; k < hitsTemp.length && counter < nHits; ++k) {
            if(query.qcity.equals(indexSearcher.doc(hitsTemp[k].doc).get("cityId"))) {
                double dist = distance(Double.parseDouble(query.qlat), Double.parseDouble(indexSearcher.doc(hitsTemp[k].doc).get("lat")), Double.parseDouble(query.qlng), Double.parseDouble(indexSearcher.doc(hitsTemp[k].doc).get("lng")), 0.0, 0.0); // distance between user and POI
                if(counter == 0) {
                    max = hitsTemp[k].score;
                    min = hitsTemp[k].score;
                    maxDist = dist;
                    minDist = dist;
                }
                else {
                    if(hitsTemp[k].score < min)
                        min = hitsTemp[k].score;
                    if(hitsTemp[k].score > max)
                        max = hitsTemp[k].score;
                    if(dist < minDist)
                        minDist = dist;
                    if(dist > maxDist)
                        maxDist = dist;
                }
                //float score = ((hitsTemp[k].score - min) / (max - min)); // Normalized
                //double dist = distance(Double.parseDouble(query.qlat), Double.parseDouble(indexSearcher.doc(hitsTemp[k].doc).get("lat")), Double.parseDouble(query.qlng), Double.parseDouble(indexSearcher.doc(hitsTemp[k].doc).get("lng")), 0.0, 0.0); // distance between user and POI
                //double scoreUpdated = score * Math.exp(-Math.pow(dist, 2));
                //double scoreUpdated = (hitsTemp[k].score==0?0.0000000001f * Math.exp(-Math.pow(dist, 2)):hitsTemp[k].score * Math.exp(-Math.pow(dist, 2)));

                //ScoreDoc tempScoreDoc = new ScoreDoc(hitsTemp[k].doc, (float) scoreUpdated);
                ScoreDoc tempScoreDoc = new ScoreDoc(hitsTemp[k].doc, hitsTemp[k].score);
                hits.add(tempScoreDoc);
                counter++;
                //System.out.println("dist: " + dist + "\tScore: " + score + "\tExp: " + Math.exp(-Math.pow(dist, 2)) + "\tUpdated: " + scoreUpdated);
            }
        }
        
        int n = hits.size();
//        float[] scoreArr = new float[n];
//        double[] expArr = new double[n];
        for (int k = 0; k < n; ++k) {
            float score = (hits.get(k).score - min) / (max - min); // Normalized
//            scoreArr[k] = score;
            double dist = distance(Double.parseDouble(query.qlat), Double.parseDouble(indexSearcher.doc(hits.get(k).doc).get("lat")), Double.parseDouble(query.qlng), Double.parseDouble(indexSearcher.doc(hits.get(k).doc).get("lng")), 0.0, 0.0); // distance between user and POI
            dist = (dist - minDist) / (maxDist - minDist); // Normalized
            double exp = Math.exp(-Math.pow(dist, 2));
//            expArr[k] = exp;
            double scoreUpdated = score * exp;
            hits.get(k).score = (float) scoreUpdated;
            //System.out.println("dist: " + dist + "\tScore: " + score + "\tExp: " + exp + "\tUpdated: " + scoreUpdated);
        }
        //System.exit(1);
        Collections.sort(hits, new cmpScoreDoc());
        //hits = hits.subList(0, Math.min(hits.size(), nHits));
//        Arrays.sort(expArr);
//        for (int k = 0; k < n; ++k) {
//            System.out.println("Score: " + scoreArr[k] + "\tExp: " + expArr[n-k-1] + "\tUpdated: " + hits.get(k).score);
//        }
        
        if(counter == 0) {
            System.out.println("Nothing found!");
            return null;
        }
        else 
            return hits.toArray(new ScoreDoc[0]);
    }
    
    public ScoreDoc[] retrieveCustomized(TRECQuery query, int nHits) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
        
        List<ScoreDoc> hits = new ArrayList<>();
        //List<ScoreDoc> hits = null;
        ScoreDoc[] hitsTemp = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(indexReader.numDocs());
        Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");


        System.out.println(query.qid+ ": " +luceneQuery.toString(fieldToSearch));

        indexSearcher.search(luceneQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hitsTemp = topDocs.scoreDocs;
        
        int counter = 0;
        if(hitsTemp != null && hitsTemp.length > 0) {
            for (int i = 0; i < hitsTemp.length && counter < nHits; ++i) {
                if(query.qcity.equals(indexSearcher.doc(hitsTemp[i].doc).get("cityId"))) {
                    ScoreDoc tempScoreDoc = hitsTemp[i];
                    hits.add(tempScoreDoc);
                    counter++;
                }
            }
        }
        //System.out.println("||||||||| MAX: " + max + "\tMIN: " + min + "\tTOTAL: " + hits.length);


//        // Normalizing the scores between 0 and 1
//        if(counter > 0) {
//            float max = hits.get(0).score;
//            float min = hits.get(hits.size() - 1).score;
//            for (int k = 0; k < hits.size(); ++k) {
//                if(max - min == 0.0f)
//                    hits.get(k).score = 0.5f; // 0.5f;  // Problem: all docs with same scores
//                else
//                    hits.get(k).score = ((hits.get(k).score - min) / (max - min));
//            }            
//        }

        //System.out.println("\n|||||||||||||||||||||||||||||||||||||\n#HITS: " + hits.length + "\n|||||||||||||||||||||||||||||||||||||");


//        // Ani...
//        for (int i = 0; i < hits.length; ++i) {
//            if(hits[i].score <= 0.0) {
//                hits[i].score = 0.5f;
//            }
//        }
//        // Updating scores
//        reRankUsingKDE(hits, query);
//
//        // Sorting hits
//        Arrays.sort(hits, new cmpScoreDoc());

        
        
//        for (int i = 0; i < hits.length; ++i) {
//            System.out.println("\nHITS: " + hits[i].doc + "\t" + hits[i].score);
//            System.out.println("TopDocs: " + topDocs.scoreDocs[i].doc + "\t" + topDocs.scoreDocs[i].score + "\n");
//        }

        
//        SearchResult []results = new SearchResult[hits.length];
//        
//        for (int i = 0; i < hits.length; ++i) {
//            results[i] = new SearchResult(hits[i], indexSearcher.doc(hits[i].doc));
//        }
        
//        Arrays.sort(results, new ReRankAni());


//        for (int i = 0; i < hits.length; ++i) {
//            hits[i] = results[i].docScore;
//        }
//

//        List<NewScore> finalList = new ArrayList<>();
//        Collections.sort(finalList, new Comparator<NewScore>(){
//            @Override
//            public int compare(NewScore t, NewScore t1) {
//                return t.score>t1.score?1:t.score==t1.score?0:-1;
//            }
//        });
//        Collections.sort(hits, new Comparator<ScoreDoc>(){
//            @Override
//            public int compare(ScoreDoc t, ScoreDoc t1) {
//                return t.score>t1.score?1:t.score==t1.score?0:-1;
//            }
//        });
        //.............
        
        if(counter == 0) {
            System.out.println("Nothing found!");
            return null;
        }
        else
            return hits.toArray(new ScoreDoc[0]);
    }

    public ScoreDoc[] retrieveCustomizedTopTerms(TRECQuery query, int nHits, int nTopK) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
//        System.out.println("Query: " + query.qid+ " (" + query.nTopTerms + ") ");
//        for (int i = 0; i < Math.min(query.nTopTerms, nTopK); ++i) {
//            System.out.println(query.topTerms.get(i).term + " " + query.topTerms.get(i).weight + " " + query.topTerms.get(i).rating);
//        }
//        System.out.println();
//        System.exit(1);
        
        List<ScoreDoc> hits = new ArrayList<>();
        //List<ScoreDoc> hits = null;
        ScoreDoc[] hitsTemp = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(indexReader.numDocs());
        //Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");

        int counter = 0;
        if(query.nTopTerms >= 0) {
            BooleanQuery booleanQuery = new BooleanQuery();

            for (int i = 0; i < Math.min(query.nTopTerms, nTopK); ++i) {
//                //Query tq = new TermQuery(new Term(fieldToSearch, query.terms.get(i).term));
//                TermQuery tq = new TermQuery(new Term(fieldToSearch, query.topTerms.get(i).term));
//                tq.setBoost((float) query.topTerms.get(i).rating);
//                //tq.setBoost((float) query.topTerms.get(i).weight);
//                booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
                

                String[] terms = query.topTerms.get(i).term.split("-");
                for (int j = 0; j < terms.length; ++j) {
                    //float contextApproVal = getTermLevelContextualAppropriateness(query, terms[j]);
                    float contextApproVal = getTermLevelContextualAppropriateness_singleContextBased(query, terms[j]);
                    TermQuery tq = new TermQuery(new Term(fieldToSearch, terms[j]));
                    //tq.setBoost((float) query.topTerms.get(i).rating);
                    //tq.setBoost((float) query.topTerms.get(i).weight);
                    tq.setBoost(contextApproVal);
                    booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
                }
                
            }

            System.out.println(query.qid+ ": " +booleanQuery.toString(fieldToSearch));

            indexSearcher.search(booleanQuery, collector); // Formal query
            topDocs = collector.topDocs();
            hitsTemp = topDocs.scoreDocs;

            counter = 0;
            if(hitsTemp != null && hitsTemp.length > 0) {
                for (int i = 0; i < hitsTemp.length && counter < nHits; ++i) {
                    if(query.qcity.equals(indexSearcher.doc(hitsTemp[i].doc).get("cityId"))) {
                        ScoreDoc tempScoreDoc = hitsTemp[i];
                        hits.add(tempScoreDoc);
                        counter++;
                    }
                }
            }
        }

        //System.out.println("||||||||| MAX: " + max + "\tMIN: " + min + "\tTOTAL: " + hits.length);


//        // Normalizing the scores between 0 and 1
//        if(counter > 0) {
//            float max = hits.get(0).score;
//            float min = hits.get(hits.size() - 1).score;
//            for (int k = 0; k < hits.size(); ++k) {
//                if(max - min == 0.0f)
//                    hits.get(k).score = 0.5f; // 0.5f;  // Problem: all docs with same scores
//                else
//                    hits.get(k).score = ((hits.get(k).score - min) / (max - min));
//            }            
//        }

        //System.out.println("\n|||||||||||||||||||||||||||||||||||||\n#HITS: " + hits.length + "\n|||||||||||||||||||||||||||||||||||||");

        //.............
        
        if(counter == 0) {
            //return null;
            hitsTemp = retrieveCustomized(query, numHits);
            if(hitsTemp == null) {
                System.out.println("Nothing found!");
                return null;                
            }
            return hitsTemp;
        }
        else
            return hits.toArray(new ScoreDoc[0]);
    }
    
    public ScoreDoc[] retrieveCustomizedTopTerms1(TRECQuery query, int nHits, int nTopK) throws Exception { // First retrieved all docs with "SHOULD" clause from whole corpus then filter with city
//        System.out.println("Query: " + query.qid+ " (" + query.nTopTerms + ") ");
//        for (int i = 0; i < Math.min(query.nTopTerms, nTopK); ++i) {
//            System.out.println(query.topTerms.get(i).term + " " + query.topTerms.get(i).weight + " " + query.topTerms.get(i).rating);
//        }
//        System.out.println();
//        System.exit(1);
        
        List<ScoreDoc> hits = new ArrayList<>();
        //List<ScoreDoc> hits = null;
        ScoreDoc[] hitsTemp = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(indexReader.numDocs());
        //Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
//        System.out.println("\n-----------------------------------------------------------------------------");

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");

        int counter = 0;
        if(query.nTopTerms >= 0) {
            BooleanQuery booleanQuery = new BooleanQuery();

            for (int i = 0; i < Math.min(query.nTopTerms, nTopK); ++i) {
//                //Query tq = new TermQuery(new Term(fieldToSearch, query.terms.get(i).term));
//                TermQuery tq = new TermQuery(new Term(fieldToSearch, query.topTerms.get(i).term));
//                tq.setBoost((float) query.topTerms.get(i).rating);
//                //tq.setBoost((float) query.topTerms.get(i).weight);
//                booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
                

                String[] terms = query.topTerms.get(i).term.split("-");
                for (int j = 0; j < terms.length; ++j) {
                    //float contextApproVal = getTermLevelContextualAppropriateness(query, terms[j]);
                    //float contextApproVal = getTermLevelContextualAppropriateness_singleContextBased(query, terms[j]);
                    TermQuery tq = new TermQuery(new Term(fieldToSearch, terms[j]));
                    //tq.setBoost((float) query.topTerms.get(i).rating);
                    tq.setBoost((float) query.topTerms.get(i).weight);
                    //tq.setBoost(contextApproVal);
                    //tq.setBoost(1.0f);
                    booleanQuery.add(tq, BooleanClause.Occur.SHOULD);
                }
                
            }

            System.out.println(query.qid+ ": " +booleanQuery.toString(fieldToSearch));

            indexSearcher.search(booleanQuery, collector); // Formal query
            topDocs = collector.topDocs();
            hitsTemp = topDocs.scoreDocs;

            counter = 0;
            if(hitsTemp != null && hitsTemp.length > 0) {
                for (int i = 0; i < hitsTemp.length && counter < nHits; ++i) {
                    if(query.qcity.equals(indexSearcher.doc(hitsTemp[i].doc).get("cityId"))) {
                        ScoreDoc tempScoreDoc = hitsTemp[i];
                        hits.add(tempScoreDoc);
                        counter++;
                    }
                }
            }
        }

        //System.out.println("||||||||| MAX: " + max + "\tMIN: " + min + "\tTOTAL: " + hits.length);


//        // Normalizing the scores between 0 and 1
//        if(counter > 0) {
//            float max = hits.get(0).score;
//            float min = hits.get(hits.size() - 1).score;
//            for (int k = 0; k < hits.size(); ++k) {
//                if(max - min == 0.0f)
//                    hits.get(k).score = 0.5f; // 0.5f;  // Problem: all docs with same scores
//                else
//                    hits.get(k).score = ((hits.get(k).score - min) / (max - min));
//            }            
//        }

        //System.out.println("\n|||||||||||||||||||||||||||||||||||||\n#HITS: " + hits.length + "\n|||||||||||||||||||||||||||||||||||||");

        //.............
        
        if(counter == 0) {
            //return null;
            hitsTemp = retrieveCustomized(query, numHits);
            if(hitsTemp == null) {
                System.out.println("Nothing found!");
                return null;                
            }
            return hitsTemp;
        }
        else
            return hits.toArray(new ScoreDoc[0]);
    }

    public ScoreDoc[] retrieve(TRECQuery query) throws Exception {

        ScoreDoc[] hits = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(numHits);
        Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
        Query cityQuery = new TermQuery(new Term("cityId", query.qcity));
        Query candidateQuery = new TermQuery(new Term("qQID", query.qid));
        //Query candidateQuery = trecQueryparser.getAnalyzedQuery(query, 3); // for parsing QID only

        
//        System.out.println("Ani Query: " + query.qtitle);
        //System.out.println("Ani Query: " + query.luceneQuery);

//        System.out.println("Ani sub-queries:");
//        for (int i = 0; i < TagsClusters.size(); ++i) {
//            if(!"-1".equals(subQueries.get(i).qClass))
//                //System.out.println(i + ": " + subQueries.get(i).qClass + " (" + subQueries.get(i).qClassWeight + ")\t" + subQueries.get(i).qtitle);
//                System.out.print("(" + subQueries.get(i).qtitle + ") ");
//        }
//        System.out.println("\n-----------------------------------------------------------------------------");
        
        PhraseQuery candidateQueryPhrase = new PhraseQuery();
        candidateQueryPhrase.add(new Term("qQID", query.qid));


        BooleanQuery booleanQuery = new BooleanQuery();
        booleanQuery.add(luceneQuery, BooleanClause.Occur.SHOULD);
        booleanQuery.add(cityQuery, BooleanClause.Occur.MUST); // City matching is MUST
        //booleanQuery.add(candidateQuery, BooleanClause.Occur.MUST);
        
        //booleanQuery.add(candidateQueryPhrase, BooleanClause.Occur.MUST);


        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");


        System.out.println(query.qid+ ": " +luceneQuery.toString(fieldToSearch));

        //indexSearcher.search(luceneQuery, collector); // Formal query
        indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hits = topDocs.scoreDocs;
        
//        // Normalizing the scores between 0 and 1
//        float max = hits[0].score;
//        float min = hits[hits.length - 1].score;
//        for (int k = 0; k < hits.length; ++k) {
//            if(max - min == 0.0f)
//                hits[k].score = 1.0f; // 0.5f;  // Problem: all docs with same scores
//            else
//                hits[k].score = ((hits[k].score - min) / (max - min));
//        }
        //System.out.println("||||||||| MAX: " + max + "\tMIN: " + min + "\tTOTAL: " + hits.length);


        //System.out.println("\n|||||||||||||||||||||||||||||||||||||\n#HITS: " + hits.length + "\n|||||||||||||||||||||||||||||||||||||");


//        // Ani...
//        for (int i = 0; i < hits.length; ++i) {
//            if(hits[i].score <= 0.0) {
//                hits[i].score = 0.5f;
//            }
//        }
//        // Updating scores
//        reRankUsingKDE(hits, query);
//
//        // Sorting hits
//        Arrays.sort(hits, new cmpScoreDoc());

        
        
//        for (int i = 0; i < hits.length; ++i) {
//            System.out.println("\nHITS: " + hits[i].doc + "\t" + hits[i].score);
//            System.out.println("TopDocs: " + topDocs.scoreDocs[i].doc + "\t" + topDocs.scoreDocs[i].score + "\n");
//        }

        
//        SearchResult []results = new SearchResult[hits.length];
//        
//        for (int i = 0; i < hits.length; ++i) {
//            results[i] = new SearchResult(hits[i], indexSearcher.doc(hits[i].doc));
//        }
        
//        Arrays.sort(results, new ReRankAni());


//        for (int i = 0; i < hits.length; ++i) {
//            hits[i] = results[i].docScore;
//        }
//

//        List<NewScore> finalList = new ArrayList<>();
//        Collections.sort(finalList, new Comparator<NewScore>(){
//            @Override
//            public int compare(NewScore t, NewScore t1) {
//                return t.score>t1.score?1:t.score==t1.score?0:-1;
//            }
//        });
//        Collections.sort(hits, new Comparator<ScoreDoc>(){
//            @Override
//            public int compare(ScoreDoc t, ScoreDoc t1) {
//                return t.score>t1.score?1:t.score==t1.score?0:-1;
//            }
//        });
        //.............
        
        if(hits == null) {
            System.out.println("Nothing found");
            return null;
        }

        return hits;
    }
    
        // retrieve docs (gereal).
    public ScoreDoc[] retrieveGeneral(TRECQuery query, int nHits) throws Exception {

        ScoreDoc[] hits = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(nHits);
        Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
        
        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");

        System.out.println(query.qid+ ": " +luceneQuery.toString(fieldToSearch));

        indexSearcher.search(luceneQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hits = topDocs.scoreDocs;
        
        if(hits == null) {
            System.out.println("Nothing found");
            return null;
        }

        return hits;
    }

    // retrieve POIs from whole corpus without any restrictions
    public ScoreDoc[] retrievePOIs(TRECQuery query, int nHits) throws Exception {

        ScoreDoc[] hits = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(nHits);
        Query luceneQuery = trecQueryparser.getAnalyzedQuery(query, 1);
        Query cityQuery = new TermQuery(new Term("cityId", query.qcity));
        
        BooleanQuery booleanQuery = new BooleanQuery();
        booleanQuery.add(luceneQuery, BooleanClause.Occur.SHOULD);
        booleanQuery.add(cityQuery, BooleanClause.Occur.MUST); // City matching is MUST

        //System.out.println("||||||||||||||||||||||||||||||||||||||||||\nluceneQuery: " + luceneQuery + "\n-----------------------------------------------------\nbooleanQuery: " + booleanQuery.toString() + "\n-----------------------------------------------------\ncityQuery: " + cityQuery.toString() + "\n-----------------------------------------------------\ncandidateQuery: " + candidateQuery.toString() + "\n||||||||||||||||||||||||||||||||||||||||||\n");

        System.out.println(query.qid+ ": " +luceneQuery.toString(fieldToSearch));

        indexSearcher.search(luceneQuery, collector); // Formal query
        //indexSearcher.search(booleanQuery, collector); // Formal query AND City matching
        topDocs = collector.topDocs();
        hits = topDocs.scoreDocs;
        
        if(hits == null) {
            System.out.println("Nothing found");
            return null;
        }

        return hits;
    }
    
    public void getTopDocs() throws Exception {
        int nHits = 100; // #topDocs
        String contextTrainingFilePath = "/store/Data/TRECAdhoc/topdocsIds_TREC8.txt";
        File file = new File(contextTrainingFilePath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);
        
        ScoreDoc[] hits = null;
        for (TRECQuery query : queries) {
            hits = retrieveGeneral(query, nHits);
            
            for (int i = 0; i < hits.length; ++i) {
                int luceneDocId = hits[i].doc;
                writer.write(indexSearcher.doc(luceneDocId).get(FIELD_ID) + "\n");
            }
        }
        writer.close();
    }
    
    // For each query term it takes topK semantically similar terms from Word2vec model
    public TRECQuery getExpandedQuery(TRECQuery query, int topK) throws Exception {
        String[] termsRaw = trecQueryparser.getAnalyzedQuery(query, 1).toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");

        List<String> terms = getUniqTerms(termsRaw);
        List<String> termsExpanded = new ArrayList<>();

        int nTerms = terms.size();
        for (int i = 0; i < nTerms; ++i) {
            //termsExpanded.add(terms.get(i));
            if(getTermIndex(terms.get(i)) >= 0) {
                List<Word2vec> W2V = topkW2Vmodel(getTermIndex(terms.get(i)));
                int nW2V = W2V.size();
                for (int j = 0; j < topK && j < nW2V; ++j) {
                    termsExpanded.add(W2V.get(j).term);
                }
            }
        }
        List<String> termsExpandedUniq = getUniqTerms(termsExpanded.toArray(new String[0]));
        
        //query.qtitle = "";
        int nTermsExpandedUniq = termsExpandedUniq.size();
        for (int i = 0; i < nTermsExpandedUniq; ++i) {
            //System.out.print(termsExpandedUniq.get(i) + " ");
            query.qtitle += termsExpandedUniq.get(i) + " ";
        }
//        System.out.println();
//        System.exit(1);
        return query;
    }
    
    // For each query term it takes topK terms from user preferences based LM term selection
    public TRECQuery getExpandedQueryWithTopTerms(TRECQuery query, float lambda, int topK) throws Exception {

        //List<String> terms = new ArrayList<>();

        //terms.addAll(getTopTerms(Integer.parseInt(query.qid), lambda, topK)); // Adding terms for user preferences with missing tags
        List<String> terms = getTopTerms(Integer.parseInt(query.qid), lambda, topK); // Adding terms for user preferences with missing tags
        
        //query.qtitle = "";
        if(terms != null) {
            int nTermsExpandedUniq = terms.size();
            for (int i = 0; i < nTermsExpandedUniq; ++i) {
                //System.out.print(termsExpandedUniq.get(i) + " ");
                query.qtitle += terms.get(i) + " ";
            }            
        }
//        System.out.println();
//        System.exit(1);
        return query;
    }
    
    // Returns list of top terms from user prefernece documents (currently considering the documents with missing tags)
    public List<String> getTopTerms(int qNo, float lambda, int nTopTerms) throws Exception {
        
//        int nUserPref = userPref.size();
//        for (int i = 0; i < nUserPref; ++i) {
//            System.out.print(userPref.get(i).queryNo + "\t");
//            for (int j = 0; j < userPref.get(i).nPreference; ++j) {
//                System.out.print(userPref.get(i).docId[j] + " " + userPref.get(i).rating[j] + " ");
//            }
//            System.out.println();
//        }

        List<String> selectedTopTerms = new ArrayList<>();
        long collectionSize = collectionSizeGlobal; //getCollectionSize();
        int nUserPref = userPref.size();
        for (int i = 0; i < nUserPref; ++i) {
            if(userPref.get(i).queryNo == qNo) {
                //System.out.println(userPref.get(i).queryNo + " ||||||||||||||||||||||||||||||");
                int nPref = userPref.get(i).nPreference;
                List<TermList> topTerms = new ArrayList<>();
                for (int j = 0; j < nPref; ++j) {
                    //System.out.print(userPref.get(i).docId[j] + " " + userPref.get(i).rating[j] + " ");
                    int luceneDocId = getLuceneDocId(userPref.get(i).docId[j]); // indexSearcher.doc(luceneDocId).get("docid")
                    if(luceneDocId >= 0) {
                        int rating = userPref.get(i).rating[j];
                        topTerms.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
                        //System.out.println("Doc: " + userPref.get(i).docId[j]);
                    }
                }

                if(!topTerms.isEmpty()) {
                    List<TermList> topTermsUniq = getUniqTermList(topTerms);
                    Collections.sort(topTermsUniq, new cmpTermListWeight());

                    String terms = "";
                    //System.out.println("topTerms: " + topTerms.size() + "\ttopTermsUniq: " + topTermsUniq.size());
                    int nTopTermsUniq = topTermsUniq.size();
                    int nTerms = Math.min(nTopTermsUniq, nTopTerms);
                    for (int l = 0; l < nTerms; ++l) {
                        //System.out.print(topTermsUniq.get(l).term + " (" + topTermsUniq.get(l).weight + ")\t");
                        terms += topTermsUniq.get(l).term + " ";
                        selectedTopTerms.add(topTermsUniq.get(l).term);
                    }
                    //System.out.println("Query: " + userPref.get(i).queryNo + " " + terms);
                    //System.out.println();

                    //queries.get(i).qtitle = terms;
                    //return terms;
                    return selectedTopTerms;
                }
                else
                    //System.out.println("Query: " + userPref.get(i).queryNo + " Empty!");
                    return null;

                //System.exit(1);
                //System.out.println();

            }
        }
        return null;
    }

    // For each query Q (q1, q2, ..., qn) it takes topM terms based on KDE score from the set of topK semantically similar (Word2vec model) terms for each query term q_i i.e. m of k terms
    public TRECQuery getExpandedQueryW2VKDE(TRECQuery query, int topM, int topK) throws Exception {
        String[] termsRaw = trecQueryparser.getAnalyzedQuery(query, 1).toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");

        List<String> terms = getUniqTerms(termsRaw);
        List<String> termsExpanded = new ArrayList<>();
        List<Word2vec> observedTerms = new ArrayList<>();
        List<Word2vec> candidateTerms = new ArrayList<>();
        
        if(terms.size() <= 1)
            return query;

        int nTerms = terms.size();
        for (int i = 0; i < nTerms; ++i) {
            //termsExpanded.add(terms.get(i));
            if(getTermIndex(terms.get(i)) >= 0) {
                Word2vec temp = new Word2vec();
                temp.term = terms.get(i);
                temp.vector = convertVectorStringToFloat(W2Vmodel.get(getTermIndex(terms.get(i))));
                observedTerms.add(temp);
                
                List<Word2vec> W2V = topkW2Vmodel(getTermIndex(terms.get(i)));
                
                int nW2V = W2V.size();
                for (int j = 0; j < topK && j < nW2V; ++j) {
                    termsExpanded.add(W2V.get(j).term);
                }
            }
        }
        List<String> termsExpandedUniq = getUniqTerms(termsExpanded.toArray(new String[0]));
        
        int nObservedTerms = observedTerms.size();
        float[] weightArray = new float[nObservedTerms]; // weight array for KDE. Weights for observed terms
        for (int i = 0; i < nObservedTerms; ++i) {
            //weightArray[i] = 1.0f;  // 1.0 means equal weights. Try using tf, Cf, collection probability, tf*IDF etc.
            //weightArray[i] = (float) (getCF(observedTerms.get(i).term) * getIdf(observedTerms.get(i).term));
            //weightArray[i] = (float) (getCF(observedTerms.get(i).term));
            weightArray[i] = (float) (getCFNormalized(observedTerms.get(i).term));
        }
        
        int nTermsExpandedUniq = termsExpandedUniq.size();
        for (int i = 0; i < nTermsExpandedUniq; ++i) {
            Word2vec temp = new Word2vec();
            temp.term = termsExpandedUniq.get(i);
            temp.vector = convertVectorStringToFloat(W2Vmodel.get(getTermIndex(termsExpandedUniq.get(i))));
            temp.KDEScore = KDEScoreForTermSelect(temp.vector, observedTerms, weightArray, observedTerms.size(), 1, 1.0f);  // using sigma=1.0, h=1
            candidateTerms.add(temp);
        }
        Collections.sort(candidateTerms, new cmpW2VKDESim());
        
        //query.qtitle = "";
        int nCandidateTerms = candidateTerms.size();
        for (int i = 0; i < topM && i < nCandidateTerms; ++i) {
            //System.out.print(termsExpandedUniq.get(i) + " ");
            query.qtitle += candidateTerms.get(i).term + " ";
        }
//        System.out.println();
//        System.exit(1);
        return query;
    }
    
    public float[] convertVectorStringToFloat(String[] vector) throws Exception {
        float[] vectorFloat = new float[vector.length];
        for (int i = 1; i < vector.length; ++i) {
            vectorFloat[i-1] = Float.parseFloat(vector[i]);
        }
        return vectorFloat;
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
    
    // Returns list of unique terms ('TermList') with maximum weights
    public List<TermList> getUniqTermListMax(List<TermList> termsRaw) throws Exception {
        List<TermList> terms = new ArrayList<>();
        
        Collections.sort(termsRaw, new cmpTermListTerm());
        
        int nTerms = termsRaw.size();
        String term = termsRaw.get(0).term;
        double weight = termsRaw.get(0).weight;
        int rating = termsRaw.get(0).rating;
        for (int i = 1; i < nTerms; ++i) {
            if(!term.equals(termsRaw.get(i).term)) {
                TermList temp = new TermList();
                temp.term = term;
                temp.weight = weight;
                temp.rating = rating;
                terms.add(temp);
                term = termsRaw.get(i).term;
                weight = termsRaw.get(i).weight;
                rating = termsRaw.get(i).rating;
            }
            else {
                if(termsRaw.get(i).weight > weight)
                    weight = termsRaw.get(i).weight;
            }
        }
        TermList temp = new TermList();
        temp.term = term;
        temp.weight = weight;
        temp.rating = rating;
        terms.add(temp);

        return terms;
    }
    
    // Returns list of unique terms ('TermList') with weights summed up
    public List<TermList> getUniqTermList(List<TermList> termsRaw) throws Exception {
        List<TermList> terms = new ArrayList<>();
        
        Collections.sort(termsRaw, new cmpTermListTerm());
        
        int nTerms = termsRaw.size();
        String term = termsRaw.get(0).term;
        double weight = termsRaw.get(0).weight;
        int rating = termsRaw.get(0).rating;
        for (int i = 1; i < nTerms; ++i) {
            if(!term.equals(termsRaw.get(i).term)) {
                TermList temp = new TermList();
                temp.term = term;
                temp.weight = weight;
                temp.rating = rating;
                terms.add(temp);
                term = termsRaw.get(i).term;
                weight = termsRaw.get(i).weight;
                rating = termsRaw.get(i).rating;
            }
            else
                weight += termsRaw.get(i).weight;
        }
        TermList temp = new TermList();
        temp.term = term;
        temp.weight = weight;
        temp.rating = rating;
        terms.add(temp);

        return terms;
    }
    
    // Returns the final hits based on average scores of documents in 'nList' hits. NB: 'termsRaw' is the merged 'nList' hits
    // Sets lowest rank (maxDoc) for the doc which is not retrieved
    public List<TermList> getUniqHits(List<TermList> termsRaw, int nList) throws Exception {
        List<TermList> terms = new ArrayList<>();
        long maxDoc = indexReader.maxDoc();
        
        Collections.sort(termsRaw, new cmpTermListTerm());
        
        int count = 1;
        int nTerms = termsRaw.size();
        String term = termsRaw.get(0).term;
        double weight = termsRaw.get(0).weight;
        int rating = termsRaw.get(0).rating;
        for (int i = 1; i < nTerms; ++i) {
            if(!term.equals(termsRaw.get(i).term)) {
                if(count < nList)
                    weight += ((nList-count) * maxDoc);
                count = 1;
                TermList temp = new TermList();
                temp.term = term;
                temp.weight = weight;
                temp.rating = rating;
                terms.add(temp);
                term = termsRaw.get(i).term;
                weight = termsRaw.get(i).weight;
                rating = termsRaw.get(i).rating;
            }
            else {
                weight += termsRaw.get(i).weight;
                count++;
            }
        }
        TermList temp = new TermList();
        temp.term = term;
        temp.weight = weight;
        temp.rating = rating;
        terms.add(temp);

        nTerms = terms.size();
        for (int i = 0; i < nTerms; ++i) {  // Taking avg. rank
            terms.get(i).weight /= nList;
        }
        
        return terms;
    }
    
    // Returns the final hits based on average scores of documents in 'nList' hits. NB: 'termsRaw' is the merged 'nList' hits
    // Sets (nHits+1) rank for the doc which is not retrieved in top 'nHits' for any hits in 'nList' hits
    public List<TermList> getUniqHits1(List<TermList> termsRaw, int nList, int nHits) throws Exception {
        List<TermList> terms = new ArrayList<>();
        
        Collections.sort(termsRaw, new cmpTermListTerm());
        
        int count = 1;
        int nTerms = termsRaw.size();
        String term = termsRaw.get(0).term;
        double weight = termsRaw.get(0).weight;
        int rating = termsRaw.get(0).rating;
        for (int i = 1; i < nTerms; ++i) {
            if(!term.equals(termsRaw.get(i).term)) {
                if(count < nList)
                    weight += ((nList-count) * (nHits+1));
                count = 1;
                TermList temp = new TermList();
                temp.term = term;
                temp.weight = weight;
                temp.rating = rating;
                terms.add(temp);
                term = termsRaw.get(i).term;
                weight = termsRaw.get(i).weight;
                rating = termsRaw.get(i).rating;
            }
            else {
                weight += termsRaw.get(i).weight;
                count++;
            }
        }
        TermList temp = new TermList();
        temp.term = term;
        temp.weight = weight;
        temp.rating = rating;
        terms.add(temp);

        nTerms = terms.size();
        for (int i = 0; i < nTerms; ++i) {  // Taking avg. rank
            terms.get(i).weight /= nList;
        }
        
        return terms;
    }
    
    // Returns estimated KDE score of term x (vector representation of x) based on x_i terms where i=0, 1, ..., n-1
    // f_w(x) = 1/nh ∑w_i . K((x - x_i) / h) for i=1, 2, ..., n [weighted KDE with gaussian kernel function K(.), bandwidth h]
    public float KDEScoreForTermSelect(float[] x, List<Word2vec> xArray, float[] wArray, int n, int h, float sigma) throws Exception {
        float score = 0.0f, expPart;

        for (int i = 0; i < n; ++i) {
            expPart = (float) Math.exp(-( Math.pow(cosineSimilarity(x, xArray.get(i).vector), 2) / 2 * Math.pow(sigma, 2) * Math.pow(h, 2)));
            score += wArray[i] / (Math.sqrt(2 * Math.PI) * sigma) * expPart;
        }
        score /= n * h;
        
        return score;
    }
    
/* Calculate distance between two points in latitude and longitude taking
 * into account height difference. If you are not interested in height
 * difference pass 0.0. Uses Haversine method as its base.
 * 
 * lat1, lon1 Start point lat2, lon2 End point el1 Start altitude in meters
 * el2 End altitude in meters
 * @returns Distance in Meters
 */
    public double distance(double lat1, double lat2, double lon1,
double lon2, double el1, double el2) {

        final int R = 6371; // Radius of the earth

        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        double distance = R * c * 1000; // convert to meters

        double height = el1 - el2;

        distance = Math.pow(distance, 2) + Math.pow(height, 2);

        //return Math.sqrt(distance); // in meters
        return Math.sqrt(distance) / 1000;
    }
    
//    public void loadQueryWiseUserPrefOld (String path) throws Exception {
//        File file = new File(path);
//        FileReader fr = new FileReader(file);
//        BufferedReader br = new BufferedReader(fr);
//        String line;
//        queryWiseUserPref = new ArrayList<>();
//        queryWiseUserPrefRating = new ArrayList<>();
//        while ((line = br.readLine()) != null) {
//            String[] temp = line.split(" ");
//            String[] temp1 = new String[temp.length-1];
//            for (int i = 0; i < temp.length-1; ++i) {
//                temp1[i] = temp[i + 1];
//            }
//            String[] tempDocIds = new String[temp1.length/2 + 1];
//            int[] tempRatings = new int[temp1.length/2 + 1];
//            tempDocIds[0] = temp[0];
//            tempRatings[0] = Integer.parseInt(temp[0]);
//            for (int i = 0, j = 1; i < temp1.length; ++i) {
//                tempDocIds[j] = temp1[i];
//                tempRatings[j++] = Integer.parseInt(temp1[++i]);
//            }
//            queryWiseUserPref.add(tempDocIds);
//            queryWiseUserPrefRating.add(tempRatings);
//            //queryWiseUserPref.add(line.split(" "));
//            //Arrays.sort(TagsClustersWeight.get(i));
//        }
//        br.close();
//        //Collections.sort(queryWiseUserPref, new cmpW2VModel());
//    }

    public void loadQueryWiseUserPref (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        userPref = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String[] temp = line.split(" ");
            String[] temp1 = new String[temp.length-1];
            for (int i = 0; i < temp.length-1; ++i) {
                temp1[i] = temp[i + 1];
            }
            String[] tempDocIds = new String[temp1.length/2];
            int[] tempRatings = new int[temp1.length/2];
            tempDocIds[0] = temp[0];
            tempRatings[0] = Integer.parseInt(temp[0]);
            for (int i = 0, j = 0; i < temp1.length; ++i) {
                tempDocIds[j] = temp1[i];
                tempRatings[j++] = Integer.parseInt(temp1[++i]);
            }
            UserPreference tempUPref = new UserPreference();
            tempUPref.queryNo = Integer.parseInt(temp[0]);
            tempUPref.docId = tempDocIds;
            tempUPref.rating = tempRatings;
            tempUPref.nPreference = tempDocIds.length;
            userPref.add(tempUPref);
            
            //queryWiseUserPref.add(line.split(" "));
            //Arrays.sort(TagsClustersWeight.get(i));
        }
        br.close(); fr.close();
        //Collections.sort(queryWiseUserPref, new cmpW2VModel());
    }
    
    public void loadQueryWiseUserPref1 (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        userPref = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String[] temp = line.split(" ");
            String[] temp1 = new String[temp.length-1];
            for (int i = 0; i < temp.length-1; ++i) {
                temp1[i] = temp[i + 1];
            }
            String[] tempDocIds = new String[temp1.length/3];
            String[] tempClusterNo = new String[temp1.length/3];
            int[] tempRatings = new int[temp1.length/3];

            tempDocIds[0] = temp[0];
            tempRatings[0] = Integer.parseInt(temp[0]);
            for (int i = 0, j = 0; i < temp1.length; ++i) {
                tempDocIds[j] = temp1[i];
                tempClusterNo[j] = temp1[++i];
                tempRatings[j++] = Integer.parseInt(temp1[++i]);
            }
            UserPreference tempUPref = new UserPreference();
            tempUPref.queryNo = Integer.parseInt(temp[0]);
            tempUPref.docId = tempDocIds;
            tempUPref.clusterId = tempClusterNo;
            tempUPref.rating = tempRatings;
            tempUPref.nPreference = tempDocIds.length;
            userPref.add(tempUPref);
            
            //queryWiseUserPref.add(line.split(" "));
            //Arrays.sort(TagsClustersWeight.get(i));
        }
        br.close(); fr.close();
        //Collections.sort(queryWiseUserPref, new cmpW2VModel());        
    }
    
    // Check whether 'docID' document exists in user pref. history 'uPref' for query 'qID' (i.e. in 'uPref.docId[]'). If found, return the corresponding rating (1, 2), return -99 otherwise
    public int checkUPrefDocHistory(UserPreference uPref, int qID, String docID) throws Exception {

        for (int i = 0; i < uPref.nPreference; ++i) {
            if(docID.equals(uPref.docId[i]) && uPref.rating[i] >= 3)
                return uPref.rating[i];
        }
        return -99;
    }
    
    // Check whether 'docID' document exists in  qrels for query 'qID' (i.e. in 'qrels.get(qRelsIndex).docId[]'). If found, return the corresponding rating (1, 2), return -99 otherwise
    public int checkUPrefDoc(int qID, String docID) throws Exception {

        int qRelsIndex = getUPrefIndex(qrels, qID);
        for (int i = 0; i < qrels.get(qRelsIndex).nPreference; ++i) {
            if(docID.equals(qrels.get(qRelsIndex).docId[i]))
                return qrels.get(qRelsIndex).rating[i];
        }
        return -99;
    }

    public int getUPrefIndex(List<UserPreference> userPref, int qID) throws Exception {
        int n = userPref.size();
        for (int i = 0; i < n; ++i) {
            if(userPref.get(i).queryNo == qID)
                return i;
        }
        return -1;
    }

    public void loadQueryWiseUserPrefTags (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        userPrefTags = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String[] temp = line.split(" ");
            String[] temp1 = new String[temp.length-1];
            for (int i = 0; i < temp.length-1; ++i) {
                temp1[i] = temp[i + 1];
            }
            String[] tempDocIds = new String[temp1.length/2];
            int[] tempRatings = new int[temp1.length/2];
            tempDocIds[0] = temp[0];
            tempRatings[0] = Integer.parseInt(temp[0]);
            for (int i = 0, j = 0; i < temp1.length; ++i) {
                tempDocIds[j] = temp1[i];
                tempRatings[j++] = Integer.parseInt(temp1[++i]);
            }
            UserPreference tempUPref = new UserPreference();
            tempUPref.queryNo = Integer.parseInt(temp[0]);
            tempUPref.docId = tempDocIds;
            tempUPref.rating = tempRatings;
            tempUPref.nPreference = tempDocIds.length;
            userPrefTags.add(tempUPref);
            
            //queryWiseUserPref.add(line.split(" "));
            //Arrays.sort(TagsClustersWeight.get(i));
        }
        br.close(); fr.close();
        //Collections.sort(queryWiseUserPref, new cmpW2VModel());
    }

    public void loadQueryWiseUserPrefNegativeTags (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        userPrefNegativeTags = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String[] temp = line.split(" ");
            String[] temp1 = new String[temp.length-1];
            for (int i = 0; i < temp.length-1; ++i) {
                temp1[i] = temp[i + 1];
            }
            String[] tempDocIds = new String[temp1.length];
            tempDocIds[0] = temp[0];
            for (int i = 0, j = 0; i < temp1.length; ++i) {
                tempDocIds[j++] = temp1[i];
            }
            UserPreference tempUPref = new UserPreference();
            tempUPref.queryNo = Integer.parseInt(temp[0]);
            tempUPref.docId = tempDocIds;
            tempUPref.nPreference = tempDocIds.length;
            userPrefNegativeTags.add(tempUPref);
            
            //queryWiseUserPref.add(line.split(" "));
            //Arrays.sort(TagsClustersWeight.get(i));
        }
        br.close(); fr.close();
        //Collections.sort(queryWiseUserPref, new cmpW2VModel());
    }
    
    // Load Foursquare data (collected by Mostafa in .json format)
    public void loadFourSquareData (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        foursquareData = new ArrayList<>();
        int c = 0;
        while ((line = br.readLine()) != null) {
            JSONObject obj = new JSONObject(line);
            String ID = obj.getString("id");
            FourSquareData temp = new FourSquareData();
            temp.TRECId = ID;
            String foursquare = obj.getString("foursquare");
            if(!foursquare.equals("null")) {
                JSONObject foursquareObj = new JSONObject(foursquare);
                if (foursquareObj.optJSONArray("categories") != null) {
                    JSONArray categoriesArray = foursquareObj.getJSONArray("categories");
                    temp.categories = new ArrayList<>();
                    int nCat = categoriesArray.length();
                    for (int i = 0; i < nCat; ++i) {
                        temp.categories.add(categoriesArray.get(i).toString().replace(" ", "-").replace("-/-", " ").replace("-&amp;-", "-&-"));
                        //temp.categories.add(categoriesArray.get(i).toString().replace(" ", "-").replace("-&amp;-", "-&-"));
                    }
                    temp.nCategories = temp.categories.size();
                } else
                    temp.nCategories = 0;
            }
            else
                temp.nCategories = 0;
            
            foursquareData.add(temp);
        }
        
        Collections.sort(foursquareData, new cmpFourSquareData());
        //System.out.println("foursquareData size: " + foursquareData.size());
        //System.exit(1);
        br.close(); fr.close();
        
//        for (int i = 0; i < foursquareData.size(); ++i) {
//            if(foursquareData.get(i).nCategories > 0) {
//                System.out.print(foursquareData.get(i).TRECId);
//                for (int j = 0; j < foursquareData.get(i).nCategories; ++j) {
//                    System.out.print(" " + foursquareData.get(i).categories.get(j));
//                }
//                System.out.println();
//            }
//        }
//        System.exit(1);
    }
    
    // Load TREC-CS query (from .json)
    public void loadTECCSQueryJson (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        treccsQueryJson = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            JSONObject obj = new JSONObject(line);
            String ID = obj.getString("id");
            String body = obj.getString("body");
            JSONObject bodyObj = new JSONObject(body);
            TRECCSQuery temp = new TRECCSQuery();
            temp.qID = ID;
            temp.TRECCSTags = "";
            temp.group = bodyObj.getString("group");
            temp.trip_type = bodyObj.getString("trip_type");
            temp.duration = bodyObj.getString("duration");
            
            treccsQueryJson.add(temp);
        }
        
        Collections.sort(treccsQueryJson, new cmpTRECCSQuery());
        //System.out.println("foursquareData size: " + foursquareData.size());
        //System.exit(1);
        br.close(); fr.close();
    }
    
    public void readContextualApproTerms () throws Exception {
        String jointContextPath = "/store/Data/TREC_CS/jointContexts_sorted";
        String singleContextPath = "/store/Data/TREC_CS/singleContexts_sorted";
        //String termsPath = "/store/Data/TREC_CS/positiveTermsForJointContexts";   // contextual appro terms classified by SVM (in addition with original terms provided by Mohammad's assessors)
        //String termsPath = "/store/Data/TREC_CS/contextualApproTermsForJointContext_Mohammad";  // Only original contextual appro terms provided by Mohammad's assessors
        String termsForJointContextPath = "/store/Data/TREC_CS/contextualApproTermsForJointContext_MohammadRaw";
        
        String contextCrowdsourcePath = "/store/Data/TREC_CS/context_crowdsource_EditedTailRemoved.txt";
        String categoryCrowdsourcePath = "/store/Data/TREC_CS/category_crowdsource_EditedTailRemoved.txt";
        String scoreCrowdsourcePath = "/store/Data/TREC_CS/score_crowdsource_EditedTailRemoved.txt";
        
        contextualApproTerms = new ArrayList<>();
        
        File file1 = new File(jointContextPath);
        FileReader fr1 = new FileReader(file1);
        BufferedReader br1 = new BufferedReader(fr1);
        File file2 = new File(termsForJointContextPath);
        FileReader fr2 = new FileReader(file2);
        BufferedReader br2 = new BufferedReader(fr2);

        String line;
        int nContext = 0;
        while ((line = br1.readLine()) != null) {
            ContextualQuery temp = new ContextualQuery();
            temp.context = line;
            temp.posTags = br2.readLine().split(" ");
            contextualApproTerms.add(temp);
        }
        br1.close(); fr1.close(); br2.close(); fr2.close();
        
        
        
        file1 = new File(contextCrowdsourcePath);
        fr1 = new FileReader(file1);
        br1 = new BufferedReader(fr1);
        file2 = new File(categoryCrowdsourcePath);
        fr2 = new FileReader(file2);
        br2 = new BufferedReader(fr2);
        File file3 = new File(scoreCrowdsourcePath);
        FileReader fr3 = new FileReader(file3);
        BufferedReader br3 = new BufferedReader(fr3);
        
        String context = br1.readLine();
        String category = br2.readLine();
        float score = Float.parseFloat(br3.readLine());
        ContextualQuery temp = new ContextualQuery();
        temp.context = context;
        List<TermList> tempList =  new ArrayList<>();
        String[] termsParsed = getParsedTerms(category.split(" "));
        for (int i = 0; i < termsParsed.length; ++i) {
            String[] termsParsedParts = termsParsed[i].split("-");
            for (int j = 0; j < termsParsedParts.length; ++j) {
                TermList tempTerm = new TermList();
                tempTerm.term = termsParsedParts[j];
                tempTerm.weight = score;
                tempList.add(tempTerm);
            }
        }
        while ((line = br1.readLine()) != null) {
            category = br2.readLine();
            score = Float.parseFloat(br3.readLine());
            if(context.equals(line)) {
                termsParsed = getParsedTerms(category.split(" "));
                for (int i = 0; i < termsParsed.length; ++i) {
                    String[] termsParsedParts = termsParsed[i].split("-");
                    for (int j = 0; j < termsParsedParts.length; ++j) {
                        TermList tempTerm = new TermList();
                        tempTerm.term = termsParsedParts[j];
                        tempTerm.weight = score;
                        tempList.add(tempTerm);
                    }
                }
            }
            else {
                temp.posTagsWeighted = getUniqTermListMax(tempList);
                contextualApproTerms.add(temp);
                temp = new ContextualQuery();
                temp.context = line;
                context = line;
                tempList =  new ArrayList<>();
                termsParsed = getParsedTerms(category.split(" "));
                for (int i = 0; i < termsParsed.length; ++i) {
                    String[] termsParsedParts = termsParsed[i].split("-");
                    for (int j = 0; j < termsParsedParts.length; ++j) {
                        TermList tempTerm = new TermList();
                        tempTerm.term = termsParsedParts[j];
                        tempTerm.weight = score;
                        tempList.add(tempTerm);
                    }
                }
            }
        }
        temp.posTagsWeighted = getUniqTermListMax(tempList);
        contextualApproTerms.add(temp);

//        for (int i = 0; i < contextualApproTerms.size(); ++i) {
//            //System.out.println(contextualApproTerms.get(i).context);
//            //System.out.println(contextualApproTerms.get(i).context + "\tLen: " + contextualApproTerms.get(i).posDocs.length);
//            System.out.println(contextualApproTerms.get(i).context + "\t" + contextualApproTerms.get(i).posTags[0] + "\tLen: " + contextualApproTerms.get(i).posTags.length);
//        }
//        System.exit(1);

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
        try {
            trecQueryparser.getAnalyzedQuery(query, 1);
            return query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").replace(" ", "-").replace("--", "-");
        }
        catch (Exception e) {
            return "NULL";
        }
    }
    
    public String[] getParsedTerms (String[] terms) throws Exception {
        String[] termsParsed = new String[terms.length];
        for (int i = 0; i < terms.length; ++i) {
            termsParsed[i] = parsedTerm(terms[i]);
        }
        return termsParsed;
    }
    
    // Returns the contextual appropriateness (fine-grained) score (max) of term 'term' for the query 'query' (joint context)
    public float getTermLevelContextualAppropriateness (TRECQuery query, String term) throws Exception {
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        String jointContext = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        //String[] terms = getContextualApproTerms(jointContext);
        String[] terms = getParsedTerms(getContextualApproTerms(jointContext));
        
        float sim = getCosineSimilarityMultiTerms2(terms[0], term);
        float sum = sim;
        float min = sim;
        float max = sim;
        for (int j = 1; j < terms.length; ++j) {
                //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarity(tags[i], terms[j]) + "\tNormalized: " + getCosineSimilarityNormalized(tags[i], terms[j]));
            //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarityMultiTerms2(tags[i], terms[j]) + "\t");
            //sim = getCosineSimilarity(tags[i], terms[j]);
            sim = getCosineSimilarityMultiTerms2(terms[j], term);
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
    
    // Pre-filtering: filter out inappropriate tags. Keep only the tags that are contextually appropriate (fine-grained) to the given joint context
    public TRECQuery preFilterTRECTags (TRECQuery query, float cutOff) throws Exception {
        TRECQuery queryFiltererd = new TRECQuery();
        queryFiltererd.qid = query.qid;
        queryFiltererd.qtitle = query.qtitle;
        queryFiltererd.fieldToSearch = query.fieldToSearch;
        queryFiltererd.qcity = query.qcity;
        
        List<TermList> tagsList = new ArrayList<>();
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        String jointContext = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        //String[] terms = getContextualApproTerms(jointContext);
        String[] terms = getParsedTerms(getContextualApproTerms(jointContext));
        
        //trecQueryparser.getAnalyzedQuery(query, 1);
        //System.out.println(query.qid + "\t" + query.qtitle);
        //System.out.println(query.qid + "\t" + query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", ""));
        //String[] tags = query.luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split(" ");
        //String[] tags = query.qtitle.replace("-", " ").split(" ");
        String[] tagsRaw = query.qtitle.split(" ");
        String[] tags = getParsedTerms(query.qtitle.split(" "));
        
//        System.out.println("jointContext: " + jointContext);
//        System.out.println("tags: " + tags.length);
//        System.out.println("terms: " + terms.length);
        for (int i = 0; i < tags.length; ++i) {
            TermList temp = new TermList();
            //temp.term = tagsRaw[i];
            temp.term = tags[i];
            //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[0] + "\t\t" + getCosineSimilarityMultiTerms2(tags[i], terms[0]) + "\t");
            //float sim = getCosineSimilarity(tags[i], terms[0]);
            //float sim = getCosineSimilarityMultiTerms(tags[i], terms[0]);
            float sim = getCosineSimilarityMultiTerms2(tags[i], terms[0]);
            float sum = sim;
            temp.min = sim;
            temp.max = sim;
            for (int j = 1; j < terms.length; ++j) {
                //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarity(tags[i], terms[j]) + "\tNormalized: " + getCosineSimilarityNormalized(tags[i], terms[j]));
                //System.out.println(query.qid + ": " + tags[i] + "\t" + terms[j] + "\t\t" + getCosineSimilarityMultiTerms2(tags[i], terms[j]) + "\t");
                //sim = getCosineSimilarity(tags[i], terms[j]);
                sim = getCosineSimilarityMultiTerms2(tags[i], terms[j]);
                if(sim < temp.min)
                    temp.min = sim;
                if(sim > temp.max)
                    temp.max = sim;
                sum += sim;
            }
            //System.out.println("-------------------------------------");
            temp.avg = sum / terms.length;
            //temp.max = getTermLevelContextualAppropriateness(query, tags[i]);    // jointContextBased
            temp.max = getTermLevelContextualAppropriateness_singleContextBased(query, tags[i]);    // singleContextBased
            tagsList.add(temp);  
        }
        //System.exit(1);
        
        // filtering out and/or creating weighted tag/term distribution based on contextual appropriateness
        queryFiltererd.qtitle = "";
        queryFiltererd.topTerms = new ArrayList<>();
        int nTagsList = tagsList.size();
        for (int i = 0; i < nTagsList; ++i) {
            TermList temp = new TermList();
            temp.term = tagsList.get(i).term;
            temp.rating = 4;    // dummy here
            if(tagsList.get(i).max < 0.0f)
                temp.weight = 0.0f;
            else
                temp.weight = tagsList.get(i).max;
            queryFiltererd.topTerms.add(temp);
            //if(tagsList.get(i).max >= cutOff)
                queryFiltererd.qtitle += tagsList.get(i).term + " ";
        }
        queryFiltererd.nTopTerms = queryFiltererd.topTerms.size();

//        if("".equals(queryFiltererd.qtitle))    // In case no terms are contextually appropriate, keep the original terms
//            queryFiltererd.qtitle = query.qtitle;
        
        return queryFiltererd;
    }
        
    // Creates a contextual appropriateness matrix for all category - context pair based on Mohammad's crowdsource data
    public void setContextualAppropriateness () throws Exception {
        String context_crowdsourcePath = "/store/Data/TREC_CS/context_crowdsource.txt";
        String category_crowdsourcePath = "/store/Data/TREC_CS/category_crowdsource_Edited.txt";
        String score_crowdsourcePath = "/store/Data/TREC_CS/score_crowdsource.txt";
        
        String poiWiseJCApproMohammadPath = "/store/Data/TREC_CS/poiWiseJCApproMohammad";   // POI wise contextual appropriatenes (Mohammad's SVM-based) score for 13 jointContexts
        //String poiWiseJCApproMohammadPath = "/store/Data/TREC_CS/poiWiseJCApproMohammad_Max";
        
        List<String> temp = new ArrayList<>();
        File file = new File(context_crowdsourcePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        int nContext = 0;
        while ((line = br.readLine()) != null) {
            temp.add(line);
            nContext++;
        }
        br.close(); fr.close();
        
        context = new String[nContext];
        category = new String[nContext];
        contextCategoryScore = new float[nContext];
        context = temp.toArray(new String[0]);
        
        file = new File(category_crowdsourcePath);
        fr = new FileReader(file);
        br = new BufferedReader(fr);
        int i = 0;
        while ((line = br.readLine()) != null) {
            category[i++] = line;
        }
        br.close(); fr.close();
        
        file = new File(score_crowdsourcePath);
        fr = new FileReader(file);
        br = new BufferedReader(fr);
        i = 0;
        while ((line = br.readLine()) != null) {
            contextCategoryScore[i++] = Float.parseFloat(line);
        }
        br.close(); fr.close();
        
        poiWiseContextualAppropriateness = new ArrayList<>();
        file = new File(poiWiseJCApproMohammadPath);
        fr = new FileReader(file);
        br = new BufferedReader(fr);
        while ((line = br.readLine()) != null) {
            String content[] = line.split(" ");
            PoiContextualAppropriateness temp1 = new PoiContextualAppropriateness();
            temp1.TRECId = content[0];
            int nJC = (content.length-1)/2; // #jointContext = 13. Format: TRECId JC1 score1 JC2 score2 ...
            temp1.jointContext = new String[nJC];
            temp1.score = new double[nJC];
            i = 0;
            for (int j = 1; j < content.length; ++j) {
                temp1.jointContext[i] = content[j++];
                temp1.score[i++] = Float.parseFloat(content[j]);
            }
            poiWiseContextualAppropriateness.add(temp1);
        }
        br.close(); fr.close();
        Collections.sort(poiWiseContextualAppropriateness, new cmpPoiContextualAppropriateness());
        
        //ContextVsCategory = new float[nContext][nContext];  
    }
    
    // Performs set subtraction (a-b) i.e. returns a String[] with strings that belong to 'a' but not in 'b'
    public String[] subtractSets (String[] a, String[] b) throws Exception {
        List<String> temp = new ArrayList<>();
        Arrays.sort(b);
        for (int i = 0; i < a.length; ++i) {
            if(Arrays.binarySearch(b, a[i]) < 0)
                temp.add(a[i]);
        }
        if(temp.size() > 0)
            return temp.toArray(new String[0]);
        else
            return null;
    }
    
    public String[] getUniqStringArray (String[] a) throws Exception {
        List<TermList> terms = new ArrayList<>();
        List<TermList> termsUniq = new ArrayList<>();
        for (int i = 0; i < a.length; ++i) {
            TermList temp = new TermList();
            temp.term = a[i];
            temp.rating = 4; // useless
            temp.weight = 1.0; // useless
            terms.add(temp);
        }
        termsUniq = getUniqTermList(terms);
        int nTermsUniq = termsUniq.size();
        String[] b = new String[nTermsUniq];
        for (int i = 0; i < nTermsUniq; ++i) {
            b[i] = termsUniq.get(i).term;
        }
        return b;
    }
    
    public void setContextualRelevance (String posContextFilePath, int kQueries, int kTerms, int nHits, List<ContextualQuery>   contextualQueryTemp) throws Exception {
        File file = new File(posContextFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        Random rand = new Random();
        
        String contexts[] = br.readLine().split(" ");
        int i = 0;
        while ((line = br.readLine()) != null) {
            ContextualQuery temp = new ContextualQuery();
            temp.context = contexts[i++];
            String[] tempArray1 = line.split(" ");
            temp.posTags = getUniqStringArray(tempArray1);
            String[] tempArray2 = br.readLine().split(" ");
            temp.negTags = getUniqStringArray(tempArray2);
            contextualQueryTemp.add(temp);
        }
        br.close(); fr.close();
        
        // Joint context such as "Group-type:-Alone-AND-Trip-duration:-Day-trip-AND-Trip-type:-Business"
        for (i = 0; i < 4; ++i) {
            for (int j = 4; j < 8; ++j) {
                for (int k = 8; k < 11; ++k) {
                    ContextualQuery temp = new ContextualQuery();
                    temp.context = contextualQueryTemp.get(i).context + "-AND-" + contextualQueryTemp.get(j).context + "-AND-" + contextualQueryTemp.get(k).context;
//                    temp.posTags = (String[]) ArrayUtils.addAll(ArrayUtils.addAll(contextualQueryTemp.get(i).posTags, contextualQueryTemp.get(j).posTags), contextualQueryTemp.get(k).posTags);
//                    temp.negTags = (String[]) ArrayUtils.addAll(ArrayUtils.addAll(contextualQueryTemp.get(i).negTags, contextualQueryTemp.get(j).negTags), contextualQueryTemp.get(k).negTags);

                    String[] tempArray1 = getUniqStringArray((String[]) ArrayUtils.addAll(ArrayUtils.addAll(contextualQueryTemp.get(i).posTags, contextualQueryTemp.get(j).posTags), contextualQueryTemp.get(k).posTags));
                    String[] tempArray2 = getUniqStringArray((String[]) ArrayUtils.addAll(ArrayUtils.addAll(contextualQueryTemp.get(i).negTags, contextualQueryTemp.get(j).negTags), contextualQueryTemp.get(k).negTags));
                    temp.posTags = tempArray1; //subtractSets(tempArray1, tempArray2);
                    temp.negTags = tempArray2; //subtractSets(tempArray2, tempArray1);
                    
                    contextualQueryTemp.add(temp);
                }
            }
        }
        
        int nContextualQuery = contextualQueryTemp.size();
        for (i = 0; i < nContextualQuery; ++i) {    // for each context 'contextualQuery.get(i).context' such as "Trip-duration:-Night-out"
            List<TermList> posDocs = new ArrayList<>();
            List<TermList> negDocs = new ArrayList<>();
            //System.out.println("||||||||||||||||| " + contextualQuery.get(i).context);
            int randLimitPos = Math.min(kTerms, contextualQueryTemp.get(i).posTags.length);
            int randLimitNeg = Math.min(kTerms, contextualQueryTemp.get(i).negTags.length);
            for (int j = 0; j < kQueries; ++j) {
                String contextQueryPos = "", contextQueryNeg = "";
                for (int k = 0; k < randLimitPos; ++k) {
                    contextQueryPos += contextualQueryTemp.get(i).posTags[rand.nextInt(contextualQueryTemp.get(i).posTags.length)] + " ";
                    //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
                }
                for (int k = 0; k < randLimitNeg; ++k) {
                    contextQueryNeg += contextualQueryTemp.get(i).negTags[rand.nextInt(contextualQueryTemp.get(i).negTags.length)] + " ";
                    //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
                }
                //System.out.println(j + ": " + contextQuery);

                TRECQuery query = queries.get(0);
                
                // Get +ve docs
                query.qtitle = contextQueryPos;
                ScoreDoc[] hits = retrievePOIs(query, nHits);
                for (int l = 0; l < hits.length; ++l) {
                    TermList temp = new TermList();
                    temp.term = indexSearcher.doc(hits[l].doc).get(FIELD_ID);
                    temp.weight = 0.0f; // useless
                    temp.rating = 1; // useless
                    posDocs.add(temp);
                }
                
                // Get -ve docs
                query.qtitle = contextQueryNeg;
                hits = retrievePOIs(query, nHits);
                for (int l = 0; l < hits.length; ++l) {
                    TermList temp = new TermList();
                    temp.term = indexSearcher.doc(hits[l].doc).get(FIELD_ID);
                    temp.weight = 0.0f; // useless
                    temp.rating = 1; // useless
                    negDocs.add(temp);
                }
            }
            List<TermList> posDocsUniq = getUniqTermList(posDocs); // Union of all +ve docs (docIDs) retrieved for 'contextualQuery.get(i).context'
            List<TermList> negDocsUniq = getUniqTermList(negDocs); // Union of all -ve docs (docIDs) retrieved for 'contextualQuery.get(i).context'

            int nPosDocsUniq = posDocsUniq.size();
            int nNosDocsUniq = negDocsUniq.size();
            contextualQueryTemp.get(i).posDocs = new String[nPosDocsUniq];
            contextualQueryTemp.get(i).negDocs = new String[nNosDocsUniq];
            for (int l = 0; l < nPosDocsUniq; ++l) {
                contextualQueryTemp.get(i).posDocs[l] = posDocsUniq.get(l).term;
            }
            for (int l = 0; l < nNosDocsUniq; ++l) {
                contextualQueryTemp.get(i).negDocs[l] = negDocsUniq.get(l).term;
            }
        }
    }
    
    // Returns the index of the context such as "Trip-duration:-Night-out" in 'contextualQueryManual'
    public int getContextualQueryManualIndex (String context) throws Exception {
        ContextualQuery temp = new ContextualQuery();
        temp.context = context;
        return Collections.binarySearch(contextualQueryManual, temp, new cmpContextQuery());
    }
    
    // Returns the index of the context such as "Trip-duration:-Night-out" in 'contextualQuery'
    public int getContextualQueryIndex (String context) throws Exception {
        ContextualQuery temp = new ContextualQuery();
        temp.context = context;
        return Collections.binarySearch(contextualQuery, temp, new cmpContextQuery());
    }
    
    // Read training data. For each context (e.g. "Trip-duration:-Night-out") train a  binary ("positive" and "negative") Naive Bayes classifier
    // trainingStatus:  0 training new
    //                  1 training update
    public void trainContextualRelevance (String contextualRelevanceTrainingFilePath, int nDocsTrainPos, int nDocsTrainNeg, int trainingStatus) throws Exception {
        System.out.println("contextualRelevanceTrainingFilePath set to: " + contextualRelevanceTrainingFilePath);
        File file = new File(contextualRelevanceTrainingFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        Random rand = new Random();
                        
        // Read data
        String contexts[] = br.readLine().split(" ");
        int i = 0;
        while ((line = br.readLine()) != null) {
            ContextualQuery temp = new ContextualQuery();
            temp.context = contexts[i];
            temp.posTags = line.split(" ");
            temp.negTags = br.readLine().split(" ");
            temp.posDocs = br.readLine().split(" ");
            temp.negDocs = br.readLine().split(" ");
            if(trainingStatus == 0) {
                temp.bayes = new BayesClassifier<String, String>();
                contextualQuery.add(temp);
            }
            else {
                contextualQuery.get(i).posDocs = temp.posDocs;
                contextualQuery.get(i).negDocs = temp.negDocs;
            }
            i++;
        }
        br.close(); fr.close();

        // Training
//        System.out.print("Training POI level contextual relevance ");
//        int nContextualQuery = contextualQuery.size();
//        for (i = 0; i < nContextualQuery; ++i) { // For each context (e.g. "Trip-duration:-Night-out") train a Naive Bayes classifier
//            
//            // Training 'positive' class
//            //for (int j = 0; j < contextualQuery.get(i).posDocs.length; ++j) {
//            for (int j = 0; j < Math.min(nDocsTrainPos, contextualQuery.get(i).posDocs.length); ++j) {
//                List<TermList> topTermsPos = new ArrayList<>();
//                int luceneDocId = getLuceneDocId(contextualQuery.get(i).posDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsPos.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    String[] positiveText = null;
//                    int nTopTermsPos = topTermsPos.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsPos.get(l).term + " ";
//                    }
//                    positiveText = terms.split("\\s");
//                    contextualQuery.get(i).bayes.learn("positive", Arrays.asList(positiveText));
//                }
//            }
//            
//            // Training 'negative' class
//            //for (int j = 0; j < contextualQuery.get(i).negDocs.length; ++j) {
//            for (int j = 0; j < Math.min(nDocsTrainNeg, contextualQuery.get(i).negDocs.length); ++j) {
//                List<TermList> topTermsNeg = new ArrayList<>();
//                int luceneDocId = getLuceneDocId(contextualQuery.get(i).negDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsNeg.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    String[] negativeText = null;
//                    int nTopTermsPos = topTermsNeg.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsNeg.get(l).term + " ";
//                    }
//                    negativeText = terms.split("\\s");
//                    contextualQuery.get(i).bayes.learn("negative", Arrays.asList(negativeText));
//                }
//            }
//            System.out.print(">");
//        }
        Collections.sort(contextualQuery, new cmpContextQuery());
        System.out.println(" Completed!");
    }
    
    public void readRandomQueriesOnly (String contextualRelevanceTrainingFilePath, int nDocsTrainPos, int nDocsTrainNeg, int trainingStatus) throws Exception {
        System.out.println("contextualRelevanceTrainingFilePath set to: " + contextualRelevanceTrainingFilePath);
        File file = new File(contextualRelevanceTrainingFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        Random rand = new Random();
                        
        // Read data
        String contexts[] = br.readLine().split(" ");
        int i = 0;
        while ((line = br.readLine()) != null) {
            ContextualQuery temp = new ContextualQuery();
            temp.context = contexts[i];
            
            temp.randomQueries = new ArrayList<>();
            String qString[] = line.split(" AniSplit ");
            
            for (int j = 0; j < qString.length; ++j) {
                TRECQuery queryTemp = new TRECQuery();
                queryTemp.qid = contexts[i];
                queryTemp.fieldToSearch =  queries.get(0).fieldToSearch;
                queryTemp.qtitle = qString[j];
                temp.randomQueries.add(queryTemp);
            }

            temp.negTags = br.readLine().split(" ");
            temp.posDocs = br.readLine().split(" ");
            temp.negDocs = br.readLine().split(" ");
            if(trainingStatus == 0) {
                temp.bayes = new BayesClassifier<String, String>();
                contextualQuery.add(temp);
            }
            else {
                contextualQuery.get(i).posDocs = temp.posDocs;
                contextualQuery.get(i).negDocs = temp.negDocs;
            }
            i++;
        }
        br.close(); fr.close();

        Collections.sort(contextualQuery, new cmpContextQuery());
        System.out.println(" Completed!");
    }
    
    // Read UQV Robust (manual) query variants
    public void readManualQueryVariant() throws Exception {
        
        String path = "/store/Data/TRECAdhoc/UQVRobust_manual_queryVariants.txt";
        System.out.println("UQVFilePath set to: " + path);
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        contextualQueryManual = new ArrayList<>();

        line = br.readLine();
        String[] content = line.split("\t");
        String qID = content[0];
        String queryVariant = content[1];
        List<TRECQuery> randomQueriesTemp = new ArrayList<>();
        TRECQuery queryTemp = new TRECQuery();
        queryTemp.qid = qID;
        queryTemp.fieldToSearch = queries.get(0).fieldToSearch;
        queryTemp.qtitle = queryVariant;
        randomQueriesTemp.add(queryTemp);
        
        String qIDtemp = qID;
        while ((line = br.readLine()) != null) {
            content = line.split("\t");
            qID = content[0];
            queryVariant = content[1];
            
            if(qID.equals(qIDtemp)) {
                queryTemp = new TRECQuery();
                queryTemp.qid = qID;
                queryTemp.fieldToSearch = queries.get(0).fieldToSearch;
                queryTemp.qtitle = queryVariant;
                randomQueriesTemp.add(queryTemp);
            }
            else {
                ContextualQuery contextualQueryTemp = new ContextualQuery();
                contextualQueryTemp.context = qIDtemp;
                contextualQueryTemp.randomQueries = randomQueriesTemp;
                contextualQueryManual.add(contextualQueryTemp);
                
                randomQueriesTemp = new ArrayList<>();
                queryTemp = new TRECQuery();
                queryTemp.qid = qID;
                queryTemp.fieldToSearch = queries.get(0).fieldToSearch;
                queryTemp.qtitle = queryVariant;
                randomQueriesTemp.add(queryTemp);
                
                qIDtemp = qID;
            }
        }
        ContextualQuery contextualQueryTemp = new ContextualQuery();
        contextualQueryTemp.context = qIDtemp;
        contextualQueryTemp.randomQueries = randomQueriesTemp;
        contextualQueryManual.add(contextualQueryTemp);

        Collections.sort(contextualQueryManual, new cmpContextQuery());
        
        br.close(); fr.close();
    }
    
    // Generate training data for contextual relevance and store it in file "/store/Data/TREC_CS/contextualRelevanceTraining.txt"
    public void getDataContextualRelevance () throws Exception {
        int kQueries = 10;  // #qureies
        int kTerms = 5;    // #terns in each query
        int nHits = 10;     // #top docs
        List<ContextualQuery>   contextualQueryTemp = new ArrayList<>();
                
        // Get trainig data ('context', 'posTags', 'negTags') and generate 'posDocs', 'negDocs'
        String contextFilePath = "/store/Data/TREC_CS/contextualQuery_appropriateness0.2.txt";
        setContextualRelevance(contextFilePath, kQueries, kTerms, nHits, contextualQueryTemp);
        
        String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining.txt";
        File file = new File(contextTrainingFilePath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);

        // Write training data 'context', 'posTags', 'negTags', 'posDocs' and 'negDocs'
        int nContextualQueryTemp = contextualQueryTemp.size();
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            writer.write(contextualQueryTemp.get(i).context + " ");
        }
        writer.write("\n");
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            System.out.println(contextualQueryTemp.get(i).context + " ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
            List<TermList> topTermsPos = new ArrayList<>();
            List<TermList> topTermsNeg = new ArrayList<>();
            for (int j = 0; j < contextualQueryTemp.get(i).posTags.length; ++j) {
                writer.write(contextualQueryTemp.get(i).posTags[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).negTags.length; ++j) {
                writer.write(contextualQueryTemp.get(i).negTags[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).posDocs.length; ++j) {
                writer.write(contextualQueryTemp.get(i).posDocs[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
                writer.write(contextualQueryTemp.get(i).negDocs[j] + " ");
            }
            writer.write("\n");
            
//            // Prints positive texts
//            int luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).posDocs[0]);
//            if (luceneDocId >= 0) {
//                topTermsPos.addAll(getTermsOnly(luceneDocId));
//                String terms = "";
//                int nTopTermsPos = topTermsPos.size();
//                for (int l = 0; l < nTopTermsPos; ++l) {
//                    terms += topTermsPos.get(l).term + " ";
//                }
//                writer.write(terms);
//            }
//            
//            for (int j = 1; j < contextualQueryTemp.get(i).posDocs.length; ++j) {
//                luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).posDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsPos = new ArrayList<>();
//                    topTermsPos.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    int nTopTermsPos = topTermsPos.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsPos.get(l).term + " ";
//                    }
//                    writer.write(" Ani_Text_Slpitter " + terms);
//                }
//            }
//            writer.write("\n");
//            
//            // Prints negative texts
//            luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).negDocs[0]);
//            if (luceneDocId >= 0) {
//                topTermsNeg.addAll(getTermsOnly(luceneDocId));
//                String terms = "";
//                int nTopTermsPos = topTermsNeg.size();
//                for (int l = 0; l < nTopTermsPos; ++l) {
//                    terms += topTermsNeg.get(l).term + " ";
//                }
//                writer.write(terms);
//            }
//            for (int j = 1; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
//                luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).negDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsNeg = new ArrayList<>();
//                    topTermsNeg.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    int nTopTermsPos = topTermsNeg.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsNeg.get(l).term + " ";
//                    }
//                    writer.write(" Ani_Text_Slpitter " + terms);
//                }
//            }
//            writer.write("\n");
        }
        writer.close();
    }
    
    // Returns contextual appropriateness (Mohammad's SVM-based) score of POI 'TRECId' for joint context 'jointContext'
    public double getPoiContextualAppropriatenessSVM (String TRECId, String jointContext) throws Exception {
        PoiContextualAppropriateness temp = new PoiContextualAppropriateness();
        temp.TRECId = TRECId;
        int index = Collections.binarySearch(poiWiseContextualAppropriateness, temp, new cmpPoiContextualAppropriateness());
        if(index >= 0) {
            for (int i = 0; i < poiWiseContextualAppropriateness.get(index).jointContext.length; ++i) {
                if(jointContext.equals(poiWiseContextualAppropriateness.get(index).jointContext[i])) {
                    return poiWiseContextualAppropriateness.get(index).score[i];
                }
            }
            return 0.0;
        }
        else
            return 0.0;
    }
    
    // Predict POI level contextual relevance of a POI 'luceneDocId', (using Naive Bayes classifier) for the current context 'context' as "positive" or "negative" and update 'classifiedPosNeg' accordingly with the confidence score 'cofidence'
    public Classified predictPOIlevelContextualRelevance (int luceneDocId, String context) throws Exception {
        List<TermList> topTermsCandidate = new ArrayList<>();
        Classified classified = new Classified();
        //int luceneDocId = getLuceneDocId(contextualQuery.get(i).posDocs[j]);

        topTermsCandidate.addAll(getTermsOnly(luceneDocId));
        String terms = "";
        String[] candidateText = null;
        int nTopTermsPos = topTermsCandidate.size();
        for (int l = 0; l < nTopTermsPos; ++l) {
            terms += topTermsCandidate.get(l).term + " ";
        }
        candidateText = terms.split("\\s");
        
        int contextualQueryIndex = getContextualQueryIndex(context);
        if ("positive".equals(contextualQuery.get(contextualQueryIndex).bayes.classify(Arrays.asList(candidateText)).getCategory())) {
            classified.classifiedPosNeg = "positive";
            classified.confidence = contextualQuery.get(contextualQueryIndex).bayes.classify(Arrays.asList(candidateText)).getProbability();
        }
        else {
            classified.classifiedPosNeg = "negative";
            classified.confidence = contextualQuery.get(contextualQueryIndex).bayes.classify(Arrays.asList(candidateText)).getProbability();
        }
        return classified;
    }
    
    public List<TermList> getAppropriateTermList (String context) throws Exception {
        List<TermList> termlist = new ArrayList<>();
        int nContextualAppropriateness = contextualAppropriateness.size();
        for (int i = 0; i < nContextualAppropriateness; ++i) {
            //System.out.println(contextualAppropriateness.get(i).context + "\t" + contextualAppropriateness.get(i).category + "\t" + contextualAppropriateness.get(i).score);
            if(context.equals(contextualAppropriateness.get(i).context)) {
                TermList temp = new TermList();
                temp.term = contextualAppropriateness.get(i).category;
                temp.weight = contextualAppropriateness.get(i).score;
                
                termlist.add(temp);
            }
        }
        
        return termlist;
    }
    
    public void writeDataJointContextualAppropriatenessGeneral (List<ContextualQuery> contextualQueryTemp, String contextTrainingFilePath) throws Exception {
        
        File file = new File(contextTrainingFilePath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);

        // Write training data 'context', 'posTags', 'negTags', 'posDocs' and 'negDocs'
        int nContextualQueryTemp = contextualQueryTemp.size();
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            writer.write(contextualQueryTemp.get(i).context + " ");
        }
        writer.write("\n");
        for (int i = 0; i < nContextualQueryTemp; ++i) {
//            System.out.println(contextualQueryTemp.get(i).context + " ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
//            for (int j = 0; j < contextualQueryTemp.get(i).posTags.length; ++j) {
//                writer.write(contextualQueryTemp.get(i).posTags[j] + " ");
//            }
            writer.write("dummy dummy dummy");
            writer.write("\n");
//            for (int j = 0; j < contextualQueryTemp.get(i).negTags.length; ++j) {
//                writer.write(contextualQueryTemp.get(i).negTags[j] + " ");
//            }
            writer.write("dummy dummy dummy");
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).posDocs.length; ++j) {
                writer.write(contextualQueryTemp.get(i).posDocs[j] + " ");
            }
            writer.write("\n");
//            for (int j = 0; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
//                writer.write(contextualQueryTemp.get(i).negDocs[j] + " ");
//            }
            writer.write("dummy dummy dummy");
            writer.write("\n");
            
        }
        writer.close();
    }
    
    public void writeDataRandomQueriesOnly (List<ContextualQuery> contextualQueryTemp, String contextTrainingFilePath) throws Exception {
        
        File file = new File(contextTrainingFilePath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);

        // Write training data 'context', 'posTags', 'negTags', 'posDocs' and 'negDocs'
        int nContextualQueryTemp = contextualQueryTemp.size();
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            writer.write(contextualQueryTemp.get(i).context + " ");
        }
        writer.write("\n");
        for (int i = 0; i < nContextualQueryTemp; ++i) {
//            System.out.println(contextualQueryTemp.get(i).context + " ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
//            for (int j = 0; j < contextualQueryTemp.get(i).posTags.length; ++j) {
//                writer.write(contextualQueryTemp.get(i).posTags[j] + " ");
//            }
            int nQueries = contextualQueryTemp.get(i).randomQueries.size();
            for (int j = 0; j < nQueries; ++j) {
                writer.write(contextualQueryTemp.get(i).randomQueries.get(j).qtitle + " AniSplit ");
            }
            writer.write("\n");
//            for (int j = 0; j < contextualQueryTemp.get(i).negTags.length; ++j) {
//                writer.write(contextualQueryTemp.get(i).negTags[j] + " ");
//            }
            writer.write("dummy dummy dummy");
            writer.write("\n");
            writer.write("dummy dummy dummy");
            writer.write("\n");
//            for (int j = 0; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
//                writer.write(contextualQueryTemp.get(i).negDocs[j] + " ");
//            }
            writer.write("dummy dummy dummy");
            writer.write("\n");
            
        }
        writer.close();
    }
    
    public void writeDataJointContextualAppropriateness (List<ContextualQuery> contextualQueryTemp, String contextTrainingFilePath) throws Exception {
        
//        String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining.txt";
        File file = new File(contextTrainingFilePath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);

        // Write training data 'context', 'posTags', 'negTags', 'posDocs' and 'negDocs'
        int nContextualQueryTemp = contextualQueryTemp.size();
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            writer.write(contextualQueryTemp.get(i).context + " ");
        }
        writer.write("\n");
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            System.out.println(contextualQueryTemp.get(i).context + " ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
            List<TermList> topTermsPos = new ArrayList<>();
            List<TermList> topTermsNeg = new ArrayList<>();
            for (int j = 0; j < contextualQueryTemp.get(i).posTags.length; ++j) {
                writer.write(contextualQueryTemp.get(i).posTags[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).negTags.length; ++j) {
                writer.write(contextualQueryTemp.get(i).negTags[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).posDocs.length; ++j) {
                writer.write(contextualQueryTemp.get(i).posDocs[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
                writer.write(contextualQueryTemp.get(i).negDocs[j] + " ");
            }
            writer.write("\n");
            
//            // Prints positive texts
//            int luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).posDocs[0]);
//            if (luceneDocId >= 0) {
//                topTermsPos.addAll(getTermsOnly(luceneDocId));
//                String terms = "";
//                int nTopTermsPos = topTermsPos.size();
//                for (int l = 0; l < nTopTermsPos; ++l) {
//                    terms += topTermsPos.get(l).term + " ";
//                }
//                writer.write(terms);
//            }
//            
//            for (int j = 1; j < contextualQueryTemp.get(i).posDocs.length; ++j) {
//                luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).posDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsPos = new ArrayList<>();
//                    topTermsPos.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    int nTopTermsPos = topTermsPos.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsPos.get(l).term + " ";
//                    }
//                    writer.write(" Ani_Text_Slpitter " + terms);
//                }
//            }
//            writer.write("\n");
//            
//            // Prints negative texts
//            luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).negDocs[0]);
//            if (luceneDocId >= 0) {
//                topTermsNeg.addAll(getTermsOnly(luceneDocId));
//                String terms = "";
//                int nTopTermsPos = topTermsNeg.size();
//                for (int l = 0; l < nTopTermsPos; ++l) {
//                    terms += topTermsNeg.get(l).term + " ";
//                }
//                writer.write(terms);
//            }
//            for (int j = 1; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
//                luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).negDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsNeg = new ArrayList<>();
//                    topTermsNeg.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    int nTopTermsPos = topTermsNeg.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsNeg.get(l).term + " ";
//                    }
//                    writer.write(" Ani_Text_Slpitter " + terms);
//                }
//            }
//            writer.write("\n");
        }
        writer.close();
    }
    
    public void writeDataJointContextualAppropriatenessOLD (List<ContextualQuery> contextualQueryTemp) throws Exception {
        
        String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining.txt";
        File file = new File(contextTrainingFilePath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);

        // Write training data 'context', 'posTags', 'negTags', 'posDocs' and 'negDocs'
        int nContextualQueryTemp = contextualQueryTemp.size();
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            writer.write(contextualQueryTemp.get(i).context + " ");
        }
        writer.write("\n");
        for (int i = 0; i < nContextualQueryTemp; ++i) {
            System.out.println(contextualQueryTemp.get(i).context + " ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
            List<TermList> topTermsPos = new ArrayList<>();
            List<TermList> topTermsNeg = new ArrayList<>();
            for (int j = 0; j < contextualQueryTemp.get(i).posTags.length; ++j) {
                writer.write(contextualQueryTemp.get(i).posTags[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).negTags.length; ++j) {
                writer.write(contextualQueryTemp.get(i).negTags[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).posDocs.length; ++j) {
                writer.write(contextualQueryTemp.get(i).posDocs[j] + " ");
            }
            writer.write("\n");
            for (int j = 0; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
                writer.write(contextualQueryTemp.get(i).negDocs[j] + " ");
            }
            writer.write("\n");
            
//            // Prints positive texts
//            int luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).posDocs[0]);
//            if (luceneDocId >= 0) {
//                topTermsPos.addAll(getTermsOnly(luceneDocId));
//                String terms = "";
//                int nTopTermsPos = topTermsPos.size();
//                for (int l = 0; l < nTopTermsPos; ++l) {
//                    terms += topTermsPos.get(l).term + " ";
//                }
//                writer.write(terms);
//            }
//            
//            for (int j = 1; j < contextualQueryTemp.get(i).posDocs.length; ++j) {
//                luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).posDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsPos = new ArrayList<>();
//                    topTermsPos.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    int nTopTermsPos = topTermsPos.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsPos.get(l).term + " ";
//                    }
//                    writer.write(" Ani_Text_Slpitter " + terms);
//                }
//            }
//            writer.write("\n");
//            
//            // Prints negative texts
//            luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).negDocs[0]);
//            if (luceneDocId >= 0) {
//                topTermsNeg.addAll(getTermsOnly(luceneDocId));
//                String terms = "";
//                int nTopTermsPos = topTermsNeg.size();
//                for (int l = 0; l < nTopTermsPos; ++l) {
//                    terms += topTermsNeg.get(l).term + " ";
//                }
//                writer.write(terms);
//            }
//            for (int j = 1; j < contextualQueryTemp.get(i).negDocs.length; ++j) {
//                luceneDocId = getLuceneDocId(contextualQueryTemp.get(i).negDocs[j]);
//                if (luceneDocId >= 0) {
//                    topTermsNeg = new ArrayList<>();
//                    topTermsNeg.addAll(getTermsOnly(luceneDocId));
//                    String terms = "";
//                    int nTopTermsPos = topTermsNeg.size();
//                    for (int l = 0; l < nTopTermsPos; ++l) {
//                        terms += topTermsNeg.get(l).term + " ";
//                    }
//                    writer.write(" Ani_Text_Slpitter " + terms);
//                }
//            }
//            writer.write("\n");
        }
        writer.close();
    }
    
    public int termExist (String[] terms, String term) throws Exception {
        for (int i = 0; i < terms.length; ++i) {
            if(term.equals(terms[i]))
                return i;
        }
        return -1;
    }
    
    // Same as termExist() above, but searches between the left bound 'l' (inclusive), and the right bound 'r' (inclusive).
    public int termExist_lr (String[] terms, String term, int l, int r) throws Exception {
        int lb = Math.max(0, l);
        int rb = Math.min(terms.length-1, r);
        for (int i = lb; i <= rb; ++i) {
            if(term.equals(terms[i]))
                return i;
        }
        return -1;
    }
    
    public int intExist (int[] inArray, int a) throws Exception {
        for (int i = 0; i < inArray.length; ++i) {
            if(inArray[i] == a)
                return i;
        }
        return -1;
    }
    
    public boolean isNumeric(String str) {
        for (char c : str.toCharArray()) {
            if (!Character.isDigit(c))
                return false;
        }
        return true;
    }
    
    public boolean charOnlyString(String str) {
        for (char c : str.toCharArray()) {
            if (Character.isDigit(c))
                return false;
        }
        return true;
    }
    
    public int findProportion (double r, TermList t) throws Exception {
        if(r < t.max && r >= t.min)
            return 1;
        else
            return 0;
    }
    
    // Returns a weight biased (Roulette wheel) random index 'i' to be selected from 'inList', to select the i-th term for random query.
    public int getBiasedRandomIndex (List<TermList> inList) throws Exception {
        double r = Math.random();
        int n = inList.size();
        for (int i = 0; i < n; ++i) {
            if(findProportion(r, inList.get(i)) == 1)
                return i;
        }
        return 0;
    }
    
    // Sum normalized
    public HashMap<String, WordProbability> getNormalizedHashMap (HashMap<String, WordProbability> inHashmap) throws Exception {
        HashMap<String, WordProbability> outHashmap = new LinkedHashMap<>();
        
        float sum = 0.0f;
        for(Map.Entry<String, WordProbability> entry: inHashmap.entrySet()) {
            sum += entry.getValue().p_w_given_R;
        }
        
        for(Map.Entry<String, WordProbability> entry: inHashmap.entrySet()) {
            outHashmap.put(entry.getKey(), new WordProbability(entry.getValue().w, entry.getValue().p_w_given_R / sum));
        }
        
        return outHashmap;
    }
    
    // Sum normalized
    public List<TermList> getNormalizedTermList (List<TermList> inList) throws Exception {
        List<TermList> outList = new ArrayList<>();
        int n = inList.size();
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += inList.get(i).weight;
        }
        
        double max = 1.0;
        for (int i = 0; i < n; ++i) {
            TermList t = new TermList();
            t.term = inList.get(i).term;
            t.weight = inList.get(i).weight / sum;
            t.max = max;
            t.min = max - t.weight;
            max = t.min;
            outList.add(t);
            //inList.get(i).weight = inList.get(i).weight / sum;
            //inList.get(i).max = max;
            //inList.get(i).min = max - inList.get(i).weight;
            //max = inList.get(i).min;
        }
        
        return outList;
    }
    
    public int getRandomInteger(int min, int max) throws Exception {
        int range = max - min + 1;
        int rand = (int) (Math.random() * range) + min;
        return rand;
    }
     
     // add random 'k' query terms ('qTerms') to the query string 'query'
    public String addRandomQueryTerms (String[] qTerms, String query) throws Exception {
        int k = getRandomInteger(1, qTerms.length);    // randomly selected k query terms
        int[] randArray = new int[k];
        for (int i = 0; i < k; ++i) {
            randArray[i] = -1;
        }
        int randArrayIndex = 0;
        for (int i = 0; i < k; ++i) {
            int j = getRandomInteger(0, qTerms.length - 1);  // randomly selected j-th query term
            while (intExist(randArray, j) >= 0) {
                j = getRandomInteger(0, qTerms.length - 1);
            }
            randArray[randArrayIndex++] = j;
            query += qTerms[j] + " ";
        }
        return query;
    }
    
    public ContextualQuery getDocsByRandomQueryingWeightedGeneral_customized (ContextualQuery contextualQueryTemp, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("\nkTerms: " + kTerms);
//        System.out.println("posTagsW: " + contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("randLimitPos: " + randLimitPos);
//        System.exit(1);
        
        List<RandomQueryHits> randomQueryHitsListPos = new ArrayList<>();       // List of +ve hits
        List<TermList> posDocs = new ArrayList<>();
        ContextualQuery contextualQueryTempUpdated = new ContextualQuery();
        
        int hitsPosCounter = 0;
        for (int j = 0; j < kQueries; ++j) {
            String contextQueryPos = "";
            int[] randArray = new int[randLimitPos];
            for (int i = 0; i < randLimitPos; ++i) {
                randArray[i] = -1;
            }
            
            int randArrayIndex = 0;
            for (int k = 0; k < randLimitPos; ++k) {
                int r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);  // random with replacement
                while(intExist(randArray, r) >= 0 || termExist(contextualQueryTemp.queryTerms, contextualQueryTemp.posTagsWeighted.get(r).term) >= 0) {   // i.e. Take only term that is not already taken and not query term (query terms will be taken separately)
                    r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);
                }
                //System.out.println("Non-query term: [" + contextualQueryTemp.posTagsWeighted.get(r).term + "]");
                randArray[randArrayIndex++] = r;
                //contextQueryPos += contextualQueryTemp.posTags[rand.nextInt(contextualQueryTemp.posTags.length)] + " ";
                contextQueryPos += contextualQueryTemp.posTagsWeighted.get(r).term + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            
//            System.out.println("-----------------------------");
//            for (int k = 0; k < contextualQueryTemp.queryTerms.length; ++k) {
//                System.out.println("Query term: [" + contextualQueryTemp.queryTerms[k]+ "]");
//            }
            //contextQueryPos += contextualQueryTemp.queryTerms[0];   // Taking only one term. etao random kora jaye
            contextQueryPos = addRandomQueryTerms(contextualQueryTemp.queryTerms, contextQueryPos); // adding random 'k' query terms
            
            //System.out.println("Added: " + contextQueryPos);
            
            //TRECQuery query = queries.get(0);
            TRECQuery query = new TRECQuery();
            query.qid = queries.get(0).qid;
            query.fieldToSearch = queries.get(0).fieldToSearch;

            // Get +ve docs
            query.qtitle = contextQueryPos;
            //System.out.print("+ve\t");
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsPos = normalizeMinMax_hits(retrievePOIs(query, nHits));
            System.out.print("+ve ");
            //ScoreDoc[] hitsPos = retrieveCustomized(query, nHits);
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            ScoreDoc[] hitsPos = retrieveGeneral(query, nHits);
            if(hitsPos != null) {
                hitsPosCounter++;
                //hitsPos = normalizeMinMax_hits(hitsPos);
                //ScoreDoc[] hitsPos = retrievePOIs_candidateCities(query, nHits);
                RandomQueryHits randomQueryHitsTempPos = new RandomQueryHits();
                randomQueryHitsTempPos.hits = new ArrayList<>();
                for (int l = 0; l < hitsPos.length; ++l) {
                    TermList temp = new TermList();

                    temp.term = indexSearcher.doc(hitsPos[l].doc).get(FIELD_ID);
                    temp.weight = l;    // 'l' holo rank. er bodole score nao. last-e MinMax normalize korte hobe...
                    //temp.weight = hitsPos[l].score;

                    randomQueryHitsTempPos.hits.add(temp);
                }
                Collections.sort(randomQueryHitsTempPos.hits, new cmpTermListTerm());
                randomQueryHitsListPos.add(randomQueryHitsTempPos);
            }
        }
        
        int nRandomQueryHitsListPos = randomQueryHitsListPos.size();
        for (int i = 0; i < nRandomQueryHitsListPos; ++i) {
            //System.out.println(randomQueryHitsListPos.get(i).hits.size());
            List<TermList> tempList = new ArrayList<>();
            for (int j = 0; j < randomQueryHitsListPos.get(i).hits.size(); ++j) {
                //System.out.println(randomQueryHitsListPos.get(i).hits.get(j).term + "\t" + randomQueryHitsListPos.get(i).hits.get(j).weight);
                TermList temp = new TermList();
                temp.term = randomQueryHitsListPos.get(i).hits.get(j).term;
                temp.weight = randomQueryHitsListPos.get(i).hits.get(j).weight;
                
                tempList.add(temp);
            }
            //System.out.println("--------------------------------------------------------------------------------------------");
            posDocs.addAll(tempList);
        }
        
        contextualQueryTempUpdated = contextualQueryTemp;
        
        //System.out.println("posDocs: " + posDocs.size());
        //List<TermList> posDocsUniq = getUniqHits(posDocs, kQueries);
        List<TermList> posDocsUniq = getUniqHits(posDocs, hitsPosCounter); // There may not have 'kQueries' hits as all queries may not get any docs at all when constraind to candidate city, for instance.
        int nPosDocsUniq = posDocsUniq.size();
        //System.out.println("posDocsUniq: " + posDocsUniq.size());
        Collections.sort(posDocsUniq, new cmpTermListWeightAscending());

        int nHitsJointFinal = Math.min(nHitsJoint, nPosDocsUniq);
        contextualQueryTempUpdated.posDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.posDocs[i] = posDocsUniq.get(i).term;
        }

        return contextualQueryTempUpdated;
    }
    
    // Based on how many times a doc is retrieved
    public ContextualQuery getDocsByRandomQueryingWeightedGeneral_customized1 (ContextualQuery contextualQueryTemp, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("\nkTerms: " + kTerms);
//        System.out.println("posTagsW: " + contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("randLimitPos: " + randLimitPos);
//        System.exit(1);
        
        List<RandomQueryHits> randomQueryHitsListPos = new ArrayList<>();       // List of +ve hits
        List<TermList> posDocs = new ArrayList<>();
        ContextualQuery contextualQueryTempUpdated = new ContextualQuery();
        
        List<TermList> docIDs = new ArrayList<>();
        
        int hitsPosCounter = 0;
        for (int j = 0; j < kQueries; ++j) {
            String contextQueryPos = "";
            int[] randArray = new int[randLimitPos];
            for (int i = 0; i < randLimitPos; ++i) {
                randArray[i] = -1;
            }
            
            int randArrayIndex = 0;
            for (int k = 0; k < randLimitPos; ++k) {
                int r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);  // random with replacement
                while(intExist(randArray, r) >= 0 || termExist(contextualQueryTemp.queryTerms, contextualQueryTemp.posTagsWeighted.get(r).term) >= 0) {   // i.e. Take only term that is not already taken and not query term (query terms will be taken separately)
                    r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);
                }
                //System.out.println("Non-query term: [" + contextualQueryTemp.posTagsWeighted.get(r).term + "]");
                randArray[randArrayIndex++] = r;
                //contextQueryPos += contextualQueryTemp.posTags[rand.nextInt(contextualQueryTemp.posTags.length)] + " ";
                contextQueryPos += contextualQueryTemp.posTagsWeighted.get(r).term + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            
//            System.out.println("-----------------------------");
//            for (int k = 0; k < contextualQueryTemp.queryTerms.length; ++k) {
//                System.out.println("Query term: [" + contextualQueryTemp.queryTerms[k]+ "]");
//            }
            //contextQueryPos += contextualQueryTemp.queryTerms[0];   // Taking only one term. etao random kora jaye
            contextQueryPos = addRandomQueryTerms(contextualQueryTemp.queryTerms, contextQueryPos); // adding random 'k' query terms
            
            //System.out.println("Added: " + contextQueryPos);
            
            //TRECQuery query = queries.get(0);
            TRECQuery query = new TRECQuery();
            query.qid = queries.get(0).qid;
            query.fieldToSearch = queries.get(0).fieldToSearch;

            // Get +ve docs
            query.qtitle = contextQueryPos;
            //System.out.print("+ve\t");
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsPos = normalizeMinMax_hits(retrievePOIs(query, nHits));
            System.out.print("+ve ");
            //ScoreDoc[] hitsPos = retrieveCustomized(query, nHits);
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            ScoreDoc[] hitsPos = retrieveGeneral(query, nHits);
            if(hitsPos != null) {
                hitsPosCounter++;
                //hitsPos = normalizeMinMax_hits(hitsPos);
                //ScoreDoc[] hitsPos = retrievePOIs_candidateCities(query, nHits);
                RandomQueryHits randomQueryHitsTempPos = new RandomQueryHits();
                randomQueryHitsTempPos.hits = new ArrayList<>();
                for (int l = 0; l < hitsPos.length; ++l) {
                    TermList temp = new TermList();

                    temp.term = indexSearcher.doc(hitsPos[l].doc).get(FIELD_ID);
                    temp.weight = l;    // 'l' holo rank. er bodole score nao. last-e MinMax normalize korte hobe...
                    //temp.weight = hitsPos[l].score;

                    randomQueryHitsTempPos.hits.add(temp);
                    docIDs.add(temp);
                }
                //Collections.sort(randomQueryHitsTempPos.hits, new cmpTermListTerm());
                randomQueryHitsListPos.add(randomQueryHitsTempPos);
            }
        }
        
        List<TermList> docIDsUniq = getUniqTermList(docIDs);
        int nRandomQueryHitsListPos = randomQueryHitsListPos.size(); // hitsPosCounter = nRandomQueryHitsListPos
        int nDocIDsUniq = docIDsUniq.size();
        for (int k = 0; k < nDocIDsUniq; ++k) {
            String docID1 = docIDsUniq.get(k).term;
            int rankCount = 0;
            float avgRank;
            for (int i = 0; i < nRandomQueryHitsListPos; ++i) {
                int existFlag = 0;
                int nHitsSize = randomQueryHitsListPos.get(i).hits.size();
                for (int j = 0; j < nHitsSize; ++j) {
                    String docID2 = randomQueryHitsListPos.get(i).hits.get(j).term;
                    if(docID1.equals(docID2)) {
                        existFlag = 1;
                        //rankCount += randomQueryHitsListPos.get(i).hits.get(j).weight;
                        rankCount++;
                        break;
                    }
                }
//                if(existFlag == 0)
//                    rankCount += nHits + 1;
            }
            avgRank = (float) rankCount / nRandomQueryHitsListPos;
            docIDsUniq.get(k).weight = avgRank;
        }
        Collections.sort(docIDsUniq, new cmpTermListWeightAscending()); // for avg rank
        //Collections.sort(docIDsUniq, new cmpTermListWeight());    // for #times retrieved
        
//        for (int k = 0; k < nDocIDsUniq; ++k) {
//            System.out.println(docIDsUniq.get(k).term + "\t" + docIDsUniq.get(k).weight);
//        }
//        System.exit(1);
        
        contextualQueryTempUpdated = contextualQueryTemp;

        int nHitsJointFinal = Math.min(nHitsJoint, nDocIDsUniq);
        contextualQueryTempUpdated.posDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.posDocs[i] = docIDsUniq.get(i).term;
        }

        return contextualQueryTempUpdated;
    }
    
    public ContextualQuery getDocsByRandomQueryingWeightedGeneral_customized2 (ContextualQuery contextualQueryTemp, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("\nkTerms: " + kTerms);
//        System.out.println("posTagsW: " + contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("randLimitPos: " + randLimitPos);
//        System.exit(1);
        
        ContextualQuery contextualQueryTempUpdated = new ContextualQuery();
        
        List<TermList> docIDs = new ArrayList<>();
        
        int hitsPosCounter = 0;
        for (int j = 0; j < kQueries; ++j) {
            String contextQueryPos = "";
            //int randomKTerms = getRandomInteger(1, randLimitPos);   // instead of sampling fixed 'kTerms' terms, sampling 'randomKTerms' terms which itself is a random number
            int randomKTerms = randLimitPos;
            int[] randArray = new int[randomKTerms];
            for (int i = 0; i < randomKTerms; ++i) {
                randArray[i] = -1;
            }
            
            int randArrayIndex = 0;
            for (int k = 0; k < randomKTerms; ++k) {
                int r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);  // random with replacement
                while(intExist(randArray, r) >= 0) {
                //while(intExist(randArray, r) >= 0 || termExist(contextualQueryTemp.queryTerms, contextualQueryTemp.posTagsWeighted.get(r).term) >= 0) {   // i.e. Take only term that is not already taken and not query term (query terms will be taken separately)
                    r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);
                }
                //System.out.println("Non-query term: [" + contextualQueryTemp.posTagsWeighted.get(r).term + "]");
                randArray[randArrayIndex++] = r;
                //contextQueryPos += contextualQueryTemp.posTags[rand.nextInt(contextualQueryTemp.posTags.length)] + " ";
                contextQueryPos += contextualQueryTemp.posTagsWeighted.get(r).term + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            
//            System.out.println("-----------------------------");
//            for (int k = 0; k < contextualQueryTemp.queryTerms.length; ++k) {
//                System.out.println("Query term: [" + contextualQueryTemp.queryTerms[k]+ "]");
//            }
            //contextQueryPos += contextualQueryTemp.queryTerms[0];   // Taking only one term. etao random kora jaye
            //contextQueryPos = addRandomQueryTerms(contextualQueryTemp.queryTerms, contextQueryPos); // adding random 'k' query terms
            
            //System.out.println("Added: " + contextQueryPos);
            
            //TRECQuery query = queries.get(0);
            TRECQuery query = new TRECQuery();
            query.qid = queries.get(0).qid;
            query.fieldToSearch = queries.get(0).fieldToSearch;

            // Get +ve docs
            query.qtitle = contextQueryPos;
            //System.out.print("+ve\t");
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsPos = normalizeMinMax_hits(retrievePOIs(query, nHits));
            System.out.print("+ve ");
            //ScoreDoc[] hitsPos = retrieveCustomized(query, nHits);
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            ScoreDoc[] hitsPos = retrieveGeneral(query, nHits);
            if(hitsPos != null) {
                hitsPosCounter++;
                //hitsPos = normalizeMinMax_hits(hitsPos);
                //ScoreDoc[] hitsPos = retrievePOIs_candidateCities(query, nHits);
                for (int l = 0; l < hitsPos.length; ++l) {
                    TermList temp = new TermList();

                    temp.term = indexSearcher.doc(hitsPos[l].doc).get(FIELD_ID);
                    temp.weight = l;    // 'l' holo rank. er bodole score nao. last-e MinMax normalize korte hobe...
                    //temp.weight = hitsPos[l].score;

                    docIDs.add(temp);
                }
                //Collections.sort(randomQueryHitsTempPos.hits, new cmpTermListTerm());
            }
        }
        
        //List<TermList> docIDsUniq = getUniqTermList(docIDs);  // #times appeared based
        List<TermList> docIDsUniq = getUniqHits(docIDs, hitsPosCounter);    // avg. rank based. Sets maxDoc for non-retrieved doc
        //List<TermList> docIDsUniq = getUniqHits1(docIDs, hitsPosCounter, nHits);  // avg. rank based. Sets nHits+1 for non-retrieved doc
        int nDocIDsUniq = docIDsUniq.size();
//        for (int k = 0; k < nDocIDsUniq; ++k) {
//            docIDsUniq.get(k).weight /= hitsPosCounter;
//        }
        
        Collections.sort(docIDsUniq, new cmpTermListWeightAscending()); // for avg rank
        //Collections.sort(docIDsUniq, new cmpTermListWeight());    // for #times retrieved
        
//        System.out.println("Uniq docs: " + nDocIDsUniq);
//        for (int k = 0; k < 100; ++k) {
//            System.out.println(docIDsUniq.get(k).term + "\t" + docIDsUniq.get(k).weight);
//        }
//        System.exit(1);
        
        contextualQueryTempUpdated = contextualQueryTemp;

        int nHitsJointFinal = Math.min(nHitsJoint, nDocIDsUniq);
        contextualQueryTempUpdated.posDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.posDocs[i] = docIDsUniq.get(i).term;
        }

        return contextualQueryTempUpdated;
    }
    
    public ContextualQuery getDocs_customized2_onManualVariants (TRECQuery query, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        //int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTagsWeighted.size());
        ContextualQuery contextualQueryTemp = new ContextualQuery();
        
        List<TermList> docIDs = new ArrayList<>();
        
        int hitsPosCounter = 0;
        int qIDindex = getContextualQueryManualIndex(query.qid);
        int kQueries = contextualQueryManual.get(qIDindex).randomQueries.size();
        for (int j = 0; j < kQueries; ++j) {
            //int randomKTerms = getRandomInteger(1, randLimitPos);   // instead of sampling fixed 'kTerms' terms, sampling 'randomKTerms' terms which itself is a random number
            //int randomKTerms = randLimitPos;

            TRECQuery queryTemp = new TRECQuery();
            queryTemp.qid = query.qid;
            queryTemp.fieldToSearch = query.fieldToSearch;

            // Get +ve docs
            queryTemp.qtitle = contextualQueryManual.get(qIDindex).randomQueries.get(j).qtitle;
            //System.out.print("+ve\t");
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsPos = normalizeMinMax_hits(retrievePOIs(query, nHits));
            System.out.print("+ve ");
            //ScoreDoc[] hitsPos = retrieveCustomized(query, nHits);
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            ScoreDoc[] hitsPos = retrieveGeneral(queryTemp, nHits);
            if(hitsPos != null) {
                hitsPosCounter++;
                //hitsPos = normalizeMinMax_hits(hitsPos);
                //ScoreDoc[] hitsPos = retrievePOIs_candidateCities(query, nHits);
                for (int l = 0; l < hitsPos.length; ++l) {
                    TermList temp = new TermList();

                    temp.term = indexSearcher.doc(hitsPos[l].doc).get(FIELD_ID);
                    temp.weight = 1;    // 'l' holo rank. er bodole score nao. last-e MinMax normalize korte hobe...
                    //temp.weight = hitsPos[l].score;

                    docIDs.add(temp);
                }
                //Collections.sort(randomQueryHitsTempPos.hits, new cmpTermListTerm());
            }
        }
        
        List<TermList> docIDsUniq = getUniqTermList(docIDs);  // #times appeared based
        //List<TermList> docIDsUniq = getUniqHits(docIDs, hitsPosCounter);    // avg. rank based. Sets maxDoc for non-retrieved doc
        //List<TermList> docIDsUniq = getUniqHits1(docIDs, hitsPosCounter, nHits);  // avg. rank based. Sets nHits+1 for non-retrieved doc
        int nDocIDsUniq = docIDsUniq.size();
//        for (int k = 0; k < nDocIDsUniq; ++k) {
//            docIDsUniq.get(k).weight /= hitsPosCounter;
//        }
        
        //Collections.sort(docIDsUniq, new cmpTermListWeightAscending()); // for avg rank
        Collections.sort(docIDsUniq, new cmpTermListWeight());    // for #times retrieved
        
//        System.out.println("Uniq docs: " + nDocIDsUniq);
//        for (int k = 0; k < 100; ++k) {
//            System.out.println(docIDsUniq.get(k).term + "\t" + docIDsUniq.get(k).weight);
//        }
//        System.exit(1);
        
        contextualQueryTemp.context = query.qid;

        int nHitsJointFinal = Math.min(nHitsJoint, nDocIDsUniq);
        contextualQueryTemp.posDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTemp.posDocs[i] = docIDsUniq.get(i).term;
        }

        return contextualQueryTemp;
    }
    
    public ContextualQuery getRandomQueriesOnly (TRECQuery query, ContextualQuery contextualQueryTemp, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTagsWeighted.size());
        
        ContextualQuery contextualQueryTempUpdated = new ContextualQuery();
        contextualQueryTempUpdated = contextualQueryTemp;
        
        List<TermList> docIDs = new ArrayList<>();
        contextualQueryTempUpdated.randomQueries = new ArrayList<>();
        
        TRECQuery queryTemp = new TRECQuery();
        queryTemp.qid = query.qid;
        queryTemp.fieldToSearch = query.fieldToSearch;
        trecQueryparser.getAnalyzedQuery(query, 1);
        queryTemp.qtitle = query.luceneQuery.toString(fieldToSearch);
        contextualQueryTempUpdated.randomQueries.add(queryTemp);    // adding the original query

        for (int j = 0; j < kQueries; ++j) {
            String contextQueryPos = "";
            //int randomKTerms = getRandomInteger(1, randLimitPos);   // instead of sampling fixed 'kTerms' terms, sampling 'randomKTerms' terms which itself is a random number
            int randomKTerms = randLimitPos;
            int[] randArray = new int[randomKTerms];
            for (int i = 0; i < randomKTerms; ++i) {
                randArray[i] = -1;
            }
            
            int randArrayIndex = 0;
            for (int k = 0; k < randomKTerms; ++k) {
                int r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);  // random with replacement
                while(intExist(randArray, r) >= 0) {
                //while(intExist(randArray, r) >= 0 || termExist(contextualQueryTemp.queryTerms, contextualQueryTemp.posTagsWeighted.get(r).term) >= 0) {   // i.e. Take only term that is not already taken and not query term (query terms will be taken separately)
                    r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);
                }
                //System.out.println("Non-query term: [" + contextualQueryTemp.posTagsWeighted.get(r).term + "]");
                randArray[randArrayIndex++] = r;
                //contextQueryPos += contextualQueryTemp.posTags[rand.nextInt(contextualQueryTemp.posTags.length)] + " ";
                contextQueryPos += contextualQueryTemp.posTagsWeighted.get(r).term + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            
//            System.out.println("-----------------------------");
//            for (int k = 0; k < contextualQueryTemp.queryTerms.length; ++k) {
//                System.out.println("Query term: [" + contextualQueryTemp.queryTerms[k]+ "]");
//            }
            //contextQueryPos += contextualQueryTemp.queryTerms[0];   // Taking only one term. etao random kora jaye
            //contextQueryPos = addRandomQueryTerms(contextualQueryTemp.queryTerms, contextQueryPos); // adding random 'k' query terms
            
            //System.out.println("Added: " + contextQueryPos);

            queryTemp = new TRECQuery();
            queryTemp.qid = query.qid;
            queryTemp.fieldToSearch = query.fieldToSearch;

            // Get +ve docs
            queryTemp.qtitle = contextQueryPos;
            contextualQueryTempUpdated.randomQueries.add(queryTemp);
        }
        
        return contextualQueryTempUpdated;
    }
    
    public HashMap<String, WordProbability> getAriRMHashmap (TRECQuery query, int nHits) throws Exception {
        
        ScoreDoc[] hits = null;
        //int contextualQueryIndex = getContextualQueryIndex(query.qid);
        int contextualQueryIndex = getContextualQueryManualIndex(query.qid);
        HashMap<String, WordProbability> hashmap_PwGivenR = new LinkedHashMap<>();
        
        //initializeRLM(numFeedbackDocsGlobal, numFeedbackTermsGlobal, QMIXGlobal);
        
        //int nQuery = contextualQuery.get(contextualQueryIndex).randomQueries.size();
        int nQuery = contextualQueryManual.get(contextualQueryIndex).randomQueries.size();
        int hitsPosCounter = 0;
        for (int i = 0; i < nQuery; ++i) {
            
            TRECQuery queryTemp = new TRECQuery();
            queryTemp.qid = query.qid;
            //queryTemp.qtitle = contextualQuery.get(contextualQueryIndex).randomQueries.get(i).qtitle;
            queryTemp.qtitle = contextualQueryManual.get(contextualQueryIndex).randomQueries.get(i).qtitle;
            queryTemp.fieldToSearch = query.fieldToSearch;
            ScoreDoc[] hitsTemp = retrieveGeneral(queryTemp, nHits);
            
        int kNN = 20;
        float threshold = 0.25f;
        float lambda = 0.4f;
        int nCluster = 2;
        //ScoreDoc[] hitsTemp = getClusterBasedTopDocs(queryTemp, kNN, threshold, lambda, nCluster);

            
            if(hitsTemp != null && hitsTemp.length > 0) {
                
                HashMap<String, WordProbability> hashmap_PwGivenRTemp;
                hitsPosCounter++;
                TopDocs topDocs = new TopDocs(hitsTemp.length, hitsTemp, hitsTemp[0].score);
                rlm.setFeedbackStatsDirect(topDocs, queryTemp.luceneQuery.toString(fieldToSearch).split(" "), 1);
                hashmap_PwGivenRTemp = rlm.RM3(queryTemp, topDocs);

                hashmap_PwGivenR = mergeHashMapsSum(hashmap_PwGivenR, hashmap_PwGivenRTemp);
            }
        }
        
        //initializeRLM(3, numFeedbackTermsGlobal, QMIXGlobal);

        HashMap<String, WordProbability> hashmap_PwGivenR_Avg = divideHashMap(hashmap_PwGivenR, hitsPosCounter);
        //HashMap<String, WordProbability> hashmap_PwGivenR_Avg_Topterms = getTopTermsHashMap(hashmap_PwGivenR_Avg, numFeedbackTermsGlobal);
        
//        BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_Avg, query);
//        //BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_Avg_Topterms, query);
//        System.out.println("Re-retrieving with QE");
//        System.out.println(booleanQuery.toString(fieldToSearch));
//        hits = retrieveGeneralBooleanQuery(query, booleanQuery, nHits);
//        
//        return hits;
        return hashmap_PwGivenR_Avg;
    }
    
    // Lu, Kurland, ... Relevance Modeling with Multiple Query Variations. ICTIR '19.
    public ScoreDoc[] ariRM (TRECQuery query, int nHits) throws Exception {
        
        ScoreDoc[] hits = null;
        //int contextualQueryIndex = getContextualQueryIndex(query.qid);
        int contextualQueryIndex = getContextualQueryManualIndex(query.qid);
        HashMap<String, WordProbability> hashmap_PwGivenR = new LinkedHashMap<>();
        
        //int nQuery = contextualQuery.get(contextualQueryIndex).randomQueries.size();
        int nQuery = contextualQueryManual.get(contextualQueryIndex).randomQueries.size();
        int hitsPosCounter = 0;
        for (int i = 0; i < nQuery; ++i) {
            
            TRECQuery queryTemp = new TRECQuery();
            queryTemp.qid = query.qid;
            //queryTemp.qtitle = contextualQuery.get(contextualQueryIndex).randomQueries.get(i).qtitle;
            queryTemp.qtitle = contextualQueryManual.get(contextualQueryIndex).randomQueries.get(i).qtitle;
            queryTemp.fieldToSearch = query.fieldToSearch;
            ScoreDoc[] hitsTemp = retrieveGeneral(queryTemp, nHits);
            
            if(hitsTemp != null && hitsTemp.length > 0) {
                HashMap<String, WordProbability> hashmap_PwGivenRTemp;
                hitsPosCounter++;
                TopDocs topDocs = new TopDocs(hitsTemp.length, hitsTemp, hitsTemp[0].score);
                rlm.setFeedbackStatsDirect(topDocs, queryTemp.luceneQuery.toString(fieldToSearch).split(" "), 1);
                hashmap_PwGivenRTemp = rlm.RM3(queryTemp, topDocs);

                hashmap_PwGivenR = mergeHashMapsSum(hashmap_PwGivenR, hashmap_PwGivenRTemp);
            }
        }

        HashMap<String, WordProbability> hashmap_PwGivenR_Avg = divideHashMap(hashmap_PwGivenR, hitsPosCounter);
        //HashMap<String, WordProbability> hashmap_PwGivenR_Avg_Topterms = getTopTermsHashMap(hashmap_PwGivenR_Avg, numFeedbackTermsGlobal);
        //HashMap<String, WordProbability> hashmap_PwGivenR_Avg_SumNormalized = getNormalizedHashMap(hashmap_PwGivenR_Avg);
        
        BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_Avg, query);
        //BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_Avg_Topterms, query);
        //BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_Avg_SumNormalized, query);
        System.out.println("Re-retrieving with QE");
        System.out.println(booleanQuery.toString(fieldToSearch));
        hits = retrieveGeneralBooleanQuery(query, booleanQuery, nHits);

        return hits;
    }
    
    // Lu, Kurland, ... Relevance Modeling with Multiple Query Variations. ICTIR '19.
    public ScoreDoc[] multiRM (TRECQuery query, int nHits) throws Exception {
        
        //ScoreDoc[] hitsMerged = null;
        List<MultipleRanklists> multiRankList = new ArrayList<>();
        
        //int contextualQueryIndex = getContextualQueryIndex(query.qid);
        int contextualQueryIndex = getContextualQueryManualIndex(query.qid);
        
        //int nQuery = contextualQuery.get(contextualQueryIndex).randomQueries.size();
        int nQuery = contextualQueryManual.get(contextualQueryIndex).randomQueries.size();
        int hitsPosCounter = 0;
        for (int i = 0; i < nQuery; ++i) {
            
            TRECQuery queryTemp = new TRECQuery();
            queryTemp.qid = query.qid;
            //queryTemp.qtitle = contextualQuery.get(contextualQueryIndex).randomQueries.get(i).qtitle;
            queryTemp.qtitle = contextualQueryManual.get(contextualQueryIndex).randomQueries.get(i).qtitle;
            queryTemp.fieldToSearch = query.fieldToSearch;
            ScoreDoc[] hitsTemp = retrieveGeneral(queryTemp, nHits);
            
            if(hitsTemp != null && hitsTemp.length > 0) {
                HashMap<String, WordProbability> hashmap_PwGivenRTemp;
                hitsPosCounter++;
                TopDocs topDocs = new TopDocs(hitsTemp.length, hitsTemp, hitsTemp[0].score);
                rlm.setFeedbackStatsDirect(topDocs, queryTemp.luceneQuery.toString(fieldToSearch).split(" "), 1);
                hashmap_PwGivenRTemp = rlm.RM3(queryTemp, topDocs);

                BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenRTemp, query);
                System.out.println("Re-retrieving with QE");
                System.out.println(booleanQuery.toString(fieldToSearch));
                hitsTemp = retrieveGeneralBooleanQuery(query, booleanQuery, nHits);
                
                MultipleRanklists tempMultiRankList = new MultipleRanklists();
                tempMultiRankList.hits = hitsTemp;
                tempMultiRankList.nDocs = nHits;
                tempMultiRankList.tagClass = "0";
                tempMultiRankList.weight = 0.5f;
                multiRankList.add(tempMultiRankList);
            }
        }

        return mergeRanklists(updateAvgScoreRanklists(multiRankList));
    }
    
    // Gets 'contextualQueryTemp' containing 'contextualQueryTemp.posTagsWeighted' as input and returns 'contextualQueryTempUpdated' containing 'contextualQueryTemp.posDocs'
    // Gets joint contextual appropriateness (generalized where query ~ jointContex) data. Instead of hard cutoff, it goes for biased/conditional random querying based on the weight of each tag/term.
    // Roulette wheel selection
    public ContextualQuery getDocsByRandomQueryingWeightedGeneral (ContextualQuery contextualQueryTemp, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("\nkTerms: " + kTerms);
//        System.out.println("posTagsW: " + contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("randLimitPos: " + randLimitPos);
//        System.exit(1);

        
        List<RandomQueryHits> randomQueryHitsListPos = new ArrayList<>();       // List of +ve hits
        List<TermList> posDocs = new ArrayList<>();
        ContextualQuery contextualQueryTempUpdated = new ContextualQuery();
        
        int hitsPosCounter = 0;
        for (int j = 0; j < kQueries; ++j) {
            String contextQueryPos = "";
            int[] randArray = new int[randLimitPos];
            for (int i = 0; i < randLimitPos; ++i) {
                randArray[i] = -1;
            }
            int randArrayIndex = 0;
            for (int k = 0; k < randLimitPos; ++k) {
                int r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);  // random with replacement
                while(intExist(randArray, r) >= 0) {
                    r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);
                }
                randArray[randArrayIndex++] = r;
                //contextQueryPos += contextualQueryTemp.posTags[rand.nextInt(contextualQueryTemp.posTags.length)] + " ";
                contextQueryPos += contextualQueryTemp.posTagsWeighted.get(r).term + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            
            //TRECQuery query = queries.get(0);
            TRECQuery query = new TRECQuery();
            query.qid = queries.get(0).qid;
            query.fieldToSearch = queries.get(0).fieldToSearch;

            // Get +ve docs
            query.qtitle = contextQueryPos;
            //System.out.print("+ve\t");
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsPos = normalizeMinMax_hits(retrievePOIs(query, nHits));
            System.out.print("+ve ");
            //ScoreDoc[] hitsPos = retrieveCustomized(query, nHits);
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            ScoreDoc[] hitsPos = retrieveGeneral(query, nHits);
            if(hitsPos != null && hitsPos.length > 0) {
                hitsPosCounter++;
                //hitsPos = normalizeMinMax_hits(hitsPos);
                //ScoreDoc[] hitsPos = retrievePOIs_candidateCities(query, nHits);
                RandomQueryHits randomQueryHitsTempPos = new RandomQueryHits();
                randomQueryHitsTempPos.hits = new ArrayList<>();
                for (int l = 0; l < hitsPos.length; ++l) {
                    TermList temp = new TermList();

                    temp.term = indexSearcher.doc(hitsPos[l].doc).get(FIELD_ID);
                    temp.weight = l;    // 'l' holo rank. er bodole score nao. last-e MinMax normalize korte hobe...
                    //temp.weight = hitsPos[l].score;

                    randomQueryHitsTempPos.hits.add(temp);
                }
                Collections.sort(randomQueryHitsTempPos.hits, new cmpTermListTerm());
                randomQueryHitsListPos.add(randomQueryHitsTempPos);
            }
        }
        
        int nRandomQueryHitsListPos = randomQueryHitsListPos.size();
        for (int i = 0; i < nRandomQueryHitsListPos; ++i) {
            //System.out.println(randomQueryHitsListPos.get(i).hits.size());
            List<TermList> tempList = new ArrayList<>();
            for (int j = 0; j < randomQueryHitsListPos.get(i).hits.size(); ++j) {
                //System.out.println(randomQueryHitsListPos.get(i).hits.get(j).term + "\t" + randomQueryHitsListPos.get(i).hits.get(j).weight);
                TermList temp = new TermList();
                temp.term = randomQueryHitsListPos.get(i).hits.get(j).term;
                temp.weight = randomQueryHitsListPos.get(i).hits.get(j).weight;
                
                tempList.add(temp);
            }
            //System.out.println("--------------------------------------------------------------------------------------------");
            posDocs.addAll(tempList);
        }
        
        contextualQueryTempUpdated = contextualQueryTemp;
        
        //System.out.println("posDocs: " + posDocs.size());
        //List<TermList> posDocsUniq = getUniqHits(posDocs, kQueries);
        List<TermList> posDocsUniq = getUniqHits(posDocs, hitsPosCounter); // There may not have 'kQueries' hits as all queries may not get any docs at all when constraind to candidate city, for instance.
        int nPosDocsUniq = posDocsUniq.size();
        //System.out.println("posDocsUniq: " + posDocsUniq.size());
        Collections.sort(posDocsUniq, new cmpTermListWeightAscending());

        int nHitsJointFinal = Math.min(nHitsJoint, nPosDocsUniq);
        contextualQueryTempUpdated.posDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.posDocs[i] = posDocsUniq.get(i).term;
        }

        return contextualQueryTempUpdated;
    }
    
    // Gets 'contextualQueryTemp' containing 'contextualQueryTemp.posTags' and 'contextualQueryTemp.negTags' as input and returns 'contextualQueryTempUpdated' containing 'contextualQueryTemp.posDocs' and 'contextualQueryTemp.negDocs'
    // Gets joint contextual appropriateness data. Instead of hard cutoff, it goes for biased/conditional random querying based on the weight of each tag/term.
    public ContextualQuery getDocsByRandomQueryingWeighted (ContextualQuery contextualQueryTemp, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        //int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTags.length);
        int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTagsWeighted.size());
        int randLimitNeg = Math.min(kTerms, contextualQueryTemp.negTags.length);
//        System.out.println("\nkTerms: " + kTerms);
//        System.out.println("posTags: " + contextualQueryTemp.posTags.length);
//        System.out.println("negsTags: " + contextualQueryTemp.negTags.length);
//        System.out.println("posTagsW: " + contextualQueryTemp.posTagsWeighted.size());
//        System.out.println("randLimitPos: " + randLimitPos);
//        System.out.println("randLimitNeg: " + randLimitNeg);
//        System.exit(1);

        
        List<RandomQueryHits> randomQueryHitsListPos = new ArrayList<>();       // List of +ve hits
        List<RandomQueryHits> randomQueryHitsListNeg = new ArrayList<>();       // List of -ve hits
        List<TermList> posDocs = new ArrayList<>();
        List<TermList> negDocs = new ArrayList<>();
        ContextualQuery contextualQueryTempUpdated = new ContextualQuery();
        
        int hitsPosCounter = 0, hitsNegCounter = 0;
        for (int j = 0; j < kQueries; ++j) {
            String contextQueryPos = "", contextQueryNeg = "";
            int[] randArray = new int[randLimitPos];
            for (int i = 0; i < randLimitPos; ++i) {
                randArray[i] = -1;
            }
            int randArrayIndex = 0;
            for (int k = 0; k < randLimitPos; ++k) {
                int r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);  // random with replacement
                while(intExist(randArray, r) >= 0) {
                    r = getBiasedRandomIndex(contextualQueryTemp.posTagsWeighted);
                }
                randArray[randArrayIndex++] = r;
                //contextQueryPos += contextualQueryTemp.posTags[rand.nextInt(contextualQueryTemp.posTags.length)] + " ";
                contextQueryPos += contextualQueryTemp.posTagsWeighted.get(r).term + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            
            for (int k = 0; k < randLimitNeg; ++k) {
                contextQueryNeg += contextualQueryTemp.negTags[rand.nextInt(contextualQueryTemp.negTags.length)] + " ";
                //contextQueryNeg += contextualQueryTemp.negTagsWeighted.get(getBiasedRandomIndex(contextualQueryTemp.negTagsWeighted)).term + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            //System.out.println(j + ": " + contextQuery);

            //TRECQuery query = queries.get(0);
            TRECQuery query = new TRECQuery();
            query.qid = queries.get(0).qid;
            query.qcity = contextualQueryTemp.city;
            query.fieldToSearch = queries.get(0).fieldToSearch;

            // Get +ve docs
            query.qtitle = contextQueryPos;
            //System.out.print("+ve\t");
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsPos = normalizeMinMax_hits(retrievePOIs(query, nHits));
            System.out.print("+ve ");
            //ScoreDoc[] hitsPos = retrieveCustomized(query, nHits);
            ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            if(hitsPos != null) {
                hitsPosCounter++;
                //hitsPos = normalizeMinMax_hits(hitsPos);
                //ScoreDoc[] hitsPos = retrievePOIs_candidateCities(query, nHits);
                RandomQueryHits randomQueryHitsTempPos = new RandomQueryHits();
                randomQueryHitsTempPos.hits = new ArrayList<>();
                for (int l = 0; l < hitsPos.length; ++l) {
                    TermList temp = new TermList();

                    temp.term = indexSearcher.doc(hitsPos[l].doc).get(FIELD_ID);
                    temp.weight = l;    // 'l' holo rank. er bodole score nao. last-e MinMax normalize korte hobe...
                    //temp.weight = hitsPos[l].score;

                    randomQueryHitsTempPos.hits.add(temp);
                }
                Collections.sort(randomQueryHitsTempPos.hits, new cmpTermListTerm());
                randomQueryHitsListPos.add(randomQueryHitsTempPos);
            }

            // Get -ve docs
            query.qtitle = contextQueryNeg;
            //System.out.print("-ve\t");
            //ScoreDoc[] hitsNeg = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsNeg = normalizeMinMax_hits(retrievePOIs(query, nHits));
            System.out.print("-ve ");
            //ScoreDoc[] hitsNeg = retrieveCustomized(query, nHits);
            ScoreDoc[] hitsNeg = retrievePOIs(query, nHits);
            if(hitsNeg != null) {
                hitsNegCounter++;
                //hitsNeg = normalizeMinMax_hits(hitsNeg);
                //ScoreDoc[] hitsNeg = retrievePOIs_candidateCities(query, nHits);
                RandomQueryHits randomQueryHitsTempNeg = new RandomQueryHits();
                randomQueryHitsTempNeg.hits = new ArrayList<>();
                for (int l = 0; l < hitsNeg.length; ++l) {
                    TermList temp = new TermList();
                    temp.term = indexSearcher.doc(hitsNeg[l].doc).get(FIELD_ID);
                    temp.weight = l;
                    //temp.weight = hitsNeg[l].score;

                    randomQueryHitsTempNeg.hits.add(temp);
                }
                Collections.sort(randomQueryHitsTempNeg.hits, new cmpTermListTerm());
                randomQueryHitsListNeg.add(randomQueryHitsTempNeg);                
            }
        }
        
        int nRandomQueryHitsListPos = randomQueryHitsListPos.size();
        for (int i = 0; i < nRandomQueryHitsListPos; ++i) {
            //System.out.println(randomQueryHitsListPos.get(i).hits.size());
            List<TermList> tempList = new ArrayList<>();
            for (int j = 0; j < randomQueryHitsListPos.get(i).hits.size(); ++j) {
                //System.out.println(randomQueryHitsListPos.get(i).hits.get(j).term + "\t" + randomQueryHitsListPos.get(i).hits.get(j).weight);
                TermList temp = new TermList();
                temp.term = randomQueryHitsListPos.get(i).hits.get(j).term;
                temp.weight = randomQueryHitsListPos.get(i).hits.get(j).weight;
                
                tempList.add(temp);
            }
            //System.out.println("--------------------------------------------------------------------------------------------");
            posDocs.addAll(tempList);
        }
        
        int nRandomQueryHitsListNeg = randomQueryHitsListNeg.size();
        for (int i = 0; i < nRandomQueryHitsListNeg; ++i) {
            //System.out.println(randomQueryHitsListNeg.get(i).hits.size());
            List<TermList> tempList = new ArrayList<>();
            for (int j = 0; j < randomQueryHitsListNeg.get(i).hits.size(); ++j) {
                //System.out.println(randomQueryHitsListPos.get(i).hits.get(j).term + "\t" + randomQueryHitsListPos.get(i).hits.get(j).weight);
                TermList temp = new TermList();
                temp.term = randomQueryHitsListNeg.get(i).hits.get(j).term;
                temp.weight = randomQueryHitsListNeg.get(i).hits.get(j).weight;
                
                tempList.add(temp);
            }
            //System.out.println("--------------------------------------------------------------------------------------------");
            negDocs.addAll(tempList);
        }
        
        contextualQueryTempUpdated = contextualQueryTemp;
        
        //System.out.println("posDocs: " + posDocs.size());
        //List<TermList> posDocsUniq = getUniqHits(posDocs, kQueries);
        List<TermList> posDocsUniq = getUniqHits(posDocs, hitsPosCounter); // There may not have 'kQueries' hits as all queries may not get any docs at all when constraind to candidate city, for instance.
        int nPosDocsUniq = posDocsUniq.size();
        //System.out.println("posDocsUniq: " + posDocsUniq.size());
        Collections.sort(posDocsUniq, new cmpTermListWeightAscending());

        //System.out.println("negDocs: " + negDocs.size());
        //List<TermList> negDocsUniq = getUniqHits(negDocs, kQueries);
        List<TermList> negDocsUniq = getUniqHits(negDocs, hitsNegCounter);
        int nNegDocsUniq = negDocsUniq.size();
        //System.out.println("negDocsDocsUniq: " + negDocsUniq.size());
        Collections.sort(negDocsUniq, new cmpTermListWeightAscending());

        int nHitsJointFinal = Math.min(nHitsJoint, nPosDocsUniq);
        contextualQueryTempUpdated.posDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.posDocs[i] = posDocsUniq.get(i).term;
        }
        
        // Gets -ve docs based on -ve tags
        nHitsJointFinal = Math.min(nHitsJoint, nNegDocsUniq);
        contextualQueryTempUpdated.negDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.negDocs[i] = negDocsUniq.get(i).term;
        }

        // Gets -ve docs from the tail of joint hits for +ve docs
//        nHitsJointFinal = Math.min(nHitsJoint, (nPosDocsUniq-nHitsJoint));
//        contextualQueryTempUpdated.negDocs = new String[nHitsJointFinal];
//        int j = 0;
//        for (int i = nPosDocsUniq-1; i >= (nPosDocsUniq-nHitsJointFinal) ; --i) {
//            contextualQueryTempUpdated.negDocs[j++] = posDocsUniq.get(i).term;
//        }

        return contextualQueryTempUpdated;
    }
    
    // Gets 'contextualQueryTemp' containing 'contextualQueryTemp.posTags' and 'contextualQueryTemp.negTags' as input and returns 'contextualQueryTempUpdated' containing 'contextualQueryTemp.posDocs' and 'contextualQueryTemp.negDocs'
    public ContextualQuery getDocsByRandomQuerying (ContextualQuery contextualQueryTemp, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        Random rand = new Random();
        int randLimitPos = Math.min(kTerms, contextualQueryTemp.posTags.length);
        int randLimitNeg = Math.min(kTerms, contextualQueryTemp.negTags.length);
        
        List<RandomQueryHits> randomQueryHitsListPos = new ArrayList<>();       // List of +ve hits
        List<RandomQueryHits> randomQueryHitsListNeg = new ArrayList<>();       // List of -ve hits
        List<TermList> posDocs = new ArrayList<>();
        List<TermList> negDocs = new ArrayList<>();
        ContextualQuery contextualQueryTempUpdated = new ContextualQuery();
        
        int hitsPosCounter = 0, hitsNegCounter = 0;
        for (int j = 0; j < kQueries; ++j) {
            String contextQueryPos = "", contextQueryNeg = "";
            for (int k = 0; k < randLimitPos; ++k) {
                contextQueryPos += contextualQueryTemp.posTags[rand.nextInt(contextualQueryTemp.posTags.length)] + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            for (int k = 0; k < randLimitNeg; ++k) {
                contextQueryNeg += contextualQueryTemp.negTags[rand.nextInt(contextualQueryTemp.negTags.length)] + " ";
                //contextQuery += rand.nextInt(contextualQuery.get(i).tags.length) + " ";
            }
            //System.out.println(j + ": " + contextQuery);

            TRECQuery query = queries.get(0);
            query.qcity = contextualQueryTemp.city;

            // Get +ve docs
            query.qtitle = contextQueryPos;
            //System.out.print("+ve\t");
            //ScoreDoc[] hitsPos = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsPos = normalizeMinMax_hits(retrievePOIs(query, nHits));
            ScoreDoc[] hitsPos = retrieveCustomized(query, nHits);
            if(hitsPos != null) {
                hitsPosCounter++;
                hitsPos = normalizeMinMax_hits(hitsPos);
                //ScoreDoc[] hitsPos = retrievePOIs_candidateCities(query, nHits);
                RandomQueryHits randomQueryHitsTempPos = new RandomQueryHits();
                randomQueryHitsTempPos.hits = new ArrayList<>();
                for (int l = 0; l < hitsPos.length; ++l) {
                    TermList temp = new TermList();

                    temp.term = indexSearcher.doc(hitsPos[l].doc).get(FIELD_ID);
                    //temp.weight = l;    // 'l' holo rank. er bodole score nao. last-e MinMax normalize korte hobe...
                    temp.weight = hitsPos[l].score;

                    randomQueryHitsTempPos.hits.add(temp);
                }
                Collections.sort(randomQueryHitsTempPos.hits, new cmpTermListTerm());
                randomQueryHitsListPos.add(randomQueryHitsTempPos);                
            }

            // Get -ve docs
            query.qtitle = contextQueryNeg;
            //System.out.print("-ve\t");
            //ScoreDoc[] hitsNeg = retrievePOIs(query, nHits);
            //ScoreDoc[] hitsNeg = normalizeMinMax_hits(retrievePOIs(query, nHits));
            ScoreDoc[] hitsNeg = retrieveCustomized(query, nHits);
            if(hitsNeg != null) {
                hitsNegCounter++;
                hitsNeg = normalizeMinMax_hits(hitsNeg);
                //ScoreDoc[] hitsNeg = retrievePOIs_candidateCities(query, nHits);
                RandomQueryHits randomQueryHitsTempNeg = new RandomQueryHits();
                randomQueryHitsTempNeg.hits = new ArrayList<>();
                for (int l = 0; l < hitsNeg.length; ++l) {
                    TermList temp = new TermList();
                    temp.term = indexSearcher.doc(hitsNeg[l].doc).get(FIELD_ID);
                    //temp.weight = l;
                    temp.weight = hitsNeg[l].score;

                    randomQueryHitsTempNeg.hits.add(temp);
                }
                Collections.sort(randomQueryHitsTempNeg.hits, new cmpTermListTerm());
                randomQueryHitsListNeg.add(randomQueryHitsTempNeg);                
            }
        }
        
        int nRandomQueryHitsListPos = randomQueryHitsListPos.size();
        for (int i = 0; i < nRandomQueryHitsListPos; ++i) {
            //System.out.println(randomQueryHitsListPos.get(i).hits.size());
            List<TermList> tempList = new ArrayList<>();
            for (int j = 0; j < randomQueryHitsListPos.get(i).hits.size(); ++j) {
                //System.out.println(randomQueryHitsListPos.get(i).hits.get(j).term + "\t" + randomQueryHitsListPos.get(i).hits.get(j).weight);
                TermList temp = new TermList();
                temp.term = randomQueryHitsListPos.get(i).hits.get(j).term;
                temp.weight = randomQueryHitsListPos.get(i).hits.get(j).weight;
                
                tempList.add(temp);
            }
            //System.out.println("--------------------------------------------------------------------------------------------");
            posDocs.addAll(tempList);
        }
        
        int nRandomQueryHitsListNeg = randomQueryHitsListNeg.size();
        for (int i = 0; i < nRandomQueryHitsListNeg; ++i) {
            //System.out.println(randomQueryHitsListNeg.get(i).hits.size());
            List<TermList> tempList = new ArrayList<>();
            for (int j = 0; j < randomQueryHitsListNeg.get(i).hits.size(); ++j) {
                //System.out.println(randomQueryHitsListPos.get(i).hits.get(j).term + "\t" + randomQueryHitsListPos.get(i).hits.get(j).weight);
                TermList temp = new TermList();
                temp.term = randomQueryHitsListNeg.get(i).hits.get(j).term;
                temp.weight = randomQueryHitsListNeg.get(i).hits.get(j).weight;
                
                tempList.add(temp);
            }
            //System.out.println("--------------------------------------------------------------------------------------------");
            negDocs.addAll(tempList);
        }
        
        contextualQueryTempUpdated = contextualQueryTemp;
        
        //System.out.println("posDocs: " + posDocs.size());
        //List<TermList> posDocsUniq = getUniqHits(posDocs, kQueries);
        List<TermList> posDocsUniq = getUniqHits(posDocs, hitsPosCounter); // There may not have 'kQueries' hits as all queries may not get any docs at all when constraind to candidate city, for instance.
        int nPosDocsUniq = posDocsUniq.size();
        //System.out.println("posDocsUniq: " + posDocsUniq.size());
        Collections.sort(posDocsUniq, new cmpTermListWeightAscending());

        //System.out.println("negDocs: " + negDocs.size());
        //List<TermList> negDocsUniq = getUniqHits(negDocs, kQueries);
        List<TermList> negDocsUniq = getUniqHits(negDocs, hitsNegCounter);
        int nNegDocsUniq = negDocsUniq.size();
        //System.out.println("negDocsDocsUniq: " + negDocsUniq.size());
        Collections.sort(negDocsUniq, new cmpTermListWeightAscending());

        int nHitsJointFinal = Math.min(nHitsJoint, nPosDocsUniq);
        contextualQueryTempUpdated.posDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.posDocs[i] = posDocsUniq.get(i).term;
        }
        
        // Gets -ve docs based on -ve tags
        nHitsJointFinal = Math.min(nHitsJoint, nNegDocsUniq);
        contextualQueryTempUpdated.negDocs = new String[nHitsJointFinal];
        for (int i = 0; i < nHitsJointFinal; ++i) {
            contextualQueryTempUpdated.negDocs[i] = negDocsUniq.get(i).term;
        }

        // Gets -ve docs from the tail of joint hits for +ve docs
//        nHitsJointFinal = Math.min(nHitsJoint, (nPosDocsUniq-nHitsJoint));
//        contextualQueryTempUpdated.negDocs = new String[nHitsJointFinal];
//        int j = 0;
//        for (int i = nPosDocsUniq-1; i >= (nPosDocsUniq-nHitsJointFinal) ; --i) {
//            contextualQueryTempUpdated.negDocs[j++] = posDocsUniq.get(i).term;
//        }

        return contextualQueryTempUpdated;
    }
    
    // get top 'kNeighbour' neighbouring (W2V) terms of the term 'term'
    public List<TermList> getTopKNeighbouringTerms (String term, int kNeighbour) throws Exception {
        int qTermIndex = getW2VTermIndex(term);

        if(qTermIndex >= 0) {
            List<Word2vec> topTermsVec = topkW2V(qTermIndex);
            List<TermList> topTerms = new ArrayList<>();
            for (int i = 0; i < kNeighbour; ++i) {  // adding top 'kNeighbour' terms pf 'term'
                TermList temp = new TermList();
                temp.term = topTermsVec.get(i).term;
                temp.weight = topTermsVec.get(i).consineScore;
                topTerms.add(temp);
            }
            
            // adding the 'term' itself
            TermList temp = new TermList();
            temp.term = term;
            temp.weight = 1.0;
            topTerms.add(temp);

            return topTerms;
        }
        
        return null;
    }
    
    // get union of top 'kNeighbour' neighbouring (W2V) terms of each query term qTerm \in query
    public List<TermList> getNeighbouringTerms (TRECQuery query, int kNeighbour) throws Exception {

        List<TermList> neighbouringTermsWeighted = new ArrayList<>();
        trecQueryparser.getAnalyzedQuery(query, 1);
        String[] qTerms = query.luceneQuery.toString(fieldToSearch).split(" ");
        System.out.println(query.luceneQuery.toString(fieldToSearch));
        
        for (int i = 0; i < qTerms.length; ++i) {
            List<TermList> temp = getTopKNeighbouringTerms(qTerms[i], kNeighbour);
            if(temp != null)
                neighbouringTermsWeighted.addAll(temp);
        }

        if(neighbouringTermsWeighted.isEmpty()) {
            System.out.println(query.qid + ": " + query.qtitle + "\tQuery terms not found in W2V.");
            System.exit(1);
        }

        //System.exit(1);
        return neighbouringTermsWeighted;
    }
    
    // getDataJointContextualAppropriateness()'s generalized implementation where jointContext ~ query and singletonContext ~ query term.
    // Generates set of docs (contextually appropritate ~ relevant to query)
    // Starting with neighbouring (to query) terms: Q={q_1, q_2, q_3} - taking union of top 'kNeighbour' terms of q_i \in Q
    public void getDataJointContextualAppropriatenessGeneral_neighbouringTerms (int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
//        int kQueries = 100;     // #queries
//        int kTerms = 5;         // #terns in each query
//        int nHits = 1000; //indexReader.maxDoc();     // #top docs for each hits
//        int nHitsJoint = 1000;   // #top docs to be selected from joint/merged list of hits' to select 'posDocs' and/or 'negDocs'
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_NeighbouringTerms="+numFeedbackTermsGlobal+".txt";
        
        List<ContextualQuery>   contextualQueryTemp = new ArrayList<>();
        
        //initializeRLM(1000, 1000, 0.4f); // int numFeedbackDocs, int numFeedbackTerms, float QMIX
        //initializeRLM(numFeedbackDocsGlobal, numFeedbackTermsGlobal, QMIXGlobal);

        int kNeighbour = numFeedbackTermsGlobal;
        //while ((line = br.readLine()) != null) {    // For each joint context
        for (TRECQuery query : queries) {   // for each query (~jointContext)
            
            List<TermList> jointTermlistWeighted = new ArrayList<>();
            jointTermlistWeighted = getNeighbouringTerms(query, kNeighbour);

            //List<TermList> jointTermlistWeightedUniq = getUniqTermList(jointTermlistWeighted);
            List<TermList> jointTermlistWeightedUniq = getUniqTermListMax(jointTermlistWeighted);
            Collections.sort(jointTermlistWeightedUniq, new cmpTermListWeight());
            jointTermlistWeightedUniq = getNormalizedTermList(jointTermlistWeightedUniq);
            
            for (int i = 0; i < jointTermlistWeightedUniq.size(); ++i) {
                System.out.println(jointTermlistWeightedUniq.get(i).term + "\t" + jointTermlistWeightedUniq.get(i).weight);
            }
            
            ContextualQuery temp = new ContextualQuery();
            temp.context = query.qid;
            temp.posTagsWeighted = jointTermlistWeightedUniq;
            temp.queryTerms = query.luceneQuery.toString(fieldToSearch).split(" ");
            System.out.println("Getting data for: " + temp.context + "\t#terms: " + temp.posTagsWeighted.size());
            
            // Gets +ve docs and -ve docs
            //ContextualQuery tempUpdated = getDocsByRandomQuerying(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeighted(temp, kQueries, kTerms, nHits, nHitsJoint);
            ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral_customized(temp, kQueries, kTerms, nHits, nHitsJoint);
            
            contextualQueryTemp.add(tempUpdated);
        }
        
        // write training data
        //writeDataJointContextualAppropriateness(contextualQueryTemp);
        writeDataJointContextualAppropriatenessGeneral(contextualQueryTemp, contextTrainingFilePath);

    }
    
    // getDataJointContextualAppropriateness()'s generalized implementation where jointContext ~ query and singletonContext ~ query term.
    // Generates set of docs (contextually appropritate ~ relevant to query)
    public void getDataJointContextualAppropriatenessGeneral (int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
//        int kQueries = 100;     // #queries
//        int kTerms = 5;         // #terns in each query
//        int nHits = 1000; //indexReader.maxDoc();     // #top docs for each hits
//        int nHitsJoint = 1000;   // #top docs to be selected from joint/merged list of hits' to select 'posDocs' and/or 'negDocs'
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_RandomKTerms_avgRankBased_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocs_FixedKTerms_cust2OverkNN_timesBased_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/RandomQueriesOnly_TRECRb_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocs_FixedKTermsOldANI_avgRankBased_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt"; // this one
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocsRandomQueryCust2_TREC7_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocsRandomQueryOnManualVariantsOld3_avgRankBased_TREC7_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        String contextTrainingFilePath = "/store/Data/TREC_CS/PRFDocs_FixedKTermsOld_avgRankBasedANIFRLM_TRECCS_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        
        List<ContextualQuery>   contextualQueryTemp = new ArrayList<>();
        
        //initializeRLM(1000, 1000, 0.4f); // int numFeedbackDocs, int numFeedbackTerms, float QMIX
        initializeRLM(numFeedbackDocsGlobal, numFeedbackTermsGlobal, QMIXGlobal);

        //while ((line = br.readLine()) != null) {    // For each joint context
        for (TRECQuery query : queries) {   // for each query (~jointContext)
            
            ScoreDoc[] hits = null;
            ScoreDoc[] hitsH = null;
            ScoreDoc[] hitsR = null;
            TopDocs topDocs;
            TopDocs topDocsH, topDocsR;
            HashMap<String, WordProbability> hashmap_PwGivenR;
            HashMap<String, WordProbability> hashmap_PwGivenR_H, hashmap_PwGivenR_R;
//            //hits = retrieveGeneral(query, numHits);
//            hits = retrieveCustomized(query, numHits);
//
//        int kNN = 20;
//        float threshold = 0.25f;
//        float lambda = 0.4f;
//        int nCluster = 2;
//        //hits = getClusterBasedTopDocs(query, kNN, threshold, lambda, nCluster);
//            topDocs = new TopDocs(hits.length, hits, hits[0].score);
//            rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
//            //hashmap_PwGivenR = rlm.RM3(query, topDocs);
//            hashmap_PwGivenR = rlm.RM3_2(query, topDocs);
//            //hashmap_PwGivenR = getAriRMHashmap(query, numHits);

//--------------------------------------------------
            UserPreference uPref = userPref.get(getUPrefIndex(userPref, Integer.parseInt(query.qid)));  // All user history docs for 'qID' query/user
hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    hitsR = retrieveCustomized(query, numHits);
                    //hitsR = getPOILevelContextualApproDocs(query, numHits);
//                    TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
//                    hitsR = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);

                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    
                    //initializeRLM(7, 20, 0.4f);
                    
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    hashmap_PwGivenR_R = rlm.RM3_2(query, topDocsR);
                    //System.exit(1);
                    
                    // Re-ranking using KL-Div
//                    hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_R, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_R, hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
//                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
//                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
//                    hashmap_PwGivenR_R = rlm.RM3_2(query, topDocsR);
                    
                    //initializeRLM(7, 30, 0.4f);

                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
                        //hashmap_PwGivenR_H = rlm.RM3Customized(query, topDocsH, uPref);
                        hashmap_PwGivenR_H = rlm.RM3Customized2(query, topDocsH, uPref);
                        
                        hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, 0.8f);
                        //hashmap_PwGivenR = hashmap_PwGivenR_H;
                    }
                    else
                        hashmap_PwGivenR = hashmap_PwGivenR_R;
//--------------------------------------------------
            
            List<TermList> jointTermlistWeighted = new ArrayList<>();
            for(Map.Entry<String, WordProbability> entry: hashmap_PwGivenR.entrySet()) {    // for each term (~singletonContext), estimated in RLM
                TermList temp = new TermList();
                temp.term = entry.getKey(); // entry.getValue().w;
                temp.weight = entry.getValue().p_w_given_R; // taking RLM weight (~averageScore)
                jointTermlistWeighted.add(temp);
            }
            Collections.sort(jointTermlistWeighted, new cmpTermListWeight());
            jointTermlistWeighted = getNormalizedTermList(jointTermlistWeighted);
            
//        System.out.println();
//        int n = jointTermlistWeighted.size();
//        for (int i = 0; i < n; ++i) {
//            //sum += jointTermlistWeighted.get(i).weight;
//            System.out.println(i + ": " + jointTermlistWeighted.get(i).term + "\t\t\t" + jointTermlistWeighted.get(i).weight + "\tMAX: " + jointTermlistWeighted.get(i).max + "\tMIN: " + jointTermlistWeighted.get(i).min);
//        }
//        System.out.println("Try 1: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 2: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 3: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 4: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 5: " + getBiasedRandomIndex(jointTermlistWeighted));
////        System.out.println("\nSUM: " + sum);
////        System.out.println("MIN: " + jointTermlistWeighted.get(n-1).weight);
////        System.out.println("MAX: " + jointTermlistWeighted.get(0).weight);
//        System.exit(1);

            
// 'jointTermlist' toiri. ete sob tags ache.
//            System.out.println("\n" + jointContext);
//            int n1=0, n2=0, n3=0, n4=0, n5=0, p1=0, p2=0, p3=0, p4=0, p5=0;
//            for (int i = 0; i < jointTermlist.size(); ++i) {
//                //System.out.println(jointTermlist.get(i).term + "\t" + jointTermlist.get(i).weight);
//                if(jointTermlist.get(i).weight >= 0.8f)
//                    p5++;
//                if(jointTermlist.get(i).weight >= 0.6f && jointTermlist.get(i).weight < 0.8f)
//                    p4++;
//                if(jointTermlist.get(i).weight >= 0.4f && jointTermlist.get(i).weight < 0.6f)
//                    p3++;
//                if(jointTermlist.get(i).weight >= 0.2f && jointTermlist.get(i).weight < 0.4f)
//                    p2++;
//                if(jointTermlist.get(i).weight >= 0.0f && jointTermlist.get(i).weight < 0.2f)
//                    p1++;
//                
//                if(jointTermlist.get(i).weight >= -0.2f && jointTermlist.get(i).weight < 0.0f)
//                    n1++;
//                if(jointTermlist.get(i).weight >= -0.4f && jointTermlist.get(i).weight < -0.2f)
//                    n2++;
//                if(jointTermlist.get(i).weight >= -0.6f && jointTermlist.get(i).weight < -0.4f)
//                    n3++;
//                if(jointTermlist.get(i).weight >= -0.8f && jointTermlist.get(i).weight < -0.6f)
//                    n4++;
//                if(jointTermlist.get(i).weight < -0.8f)
//                    n5++;
//            }
//            System.out.println("(-1.0 -0.8)\t(-0.8 -0.6)\t(-0.6 -0.4)\t(-0.4 -0.2)\t(-0.2 0.0)\t(0.0 0.2)\t(0.2 0.4)\t(0.4 0.6)\t(0.6 0.8)\t(0.8 1.0)");
//            //System.out.println(n5 + "\t" + n4 + "\t" + n3 + "\t" + n2 + "\t" + n1 + "\t" + p1 + "\t" + p2 + "\t" + p3 + "\t" + p4 + "\t" + p5);
//            System.out.println(n5 + "\t\t" + n4 + "\t\t" + n3 + "\t\t" + n2 + "\t\t" + n1 + "\t\t" + p1 + "\t\t" + p2 + "\t\t" + p3 + "\t\t" + p4 + "\t\t" + p5);
            //System.exit(1);
            
            
            ContextualQuery temp = new ContextualQuery();
            temp.context = query.qid;
            temp.posTagsWeighted = jointTermlistWeighted;
            //temp.queryTerms = query.luceneQuery.toString(fieldToSearch).split(" ");
            System.out.println("Getting data for: " + temp.context + "\t#terms: " + temp.posTagsWeighted.size());
            
            // Gets +ve docs and -ve docs
            //ContextualQuery tempUpdated = getDocsByRandomQuerying(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeighted(temp, kQueries, kTerms, nHits, nHitsJoint);
            ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral_customized(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral_customized1(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral_customized2(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getRandomQueriesOnly(query, temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocs_customized2_onManualVariants(query, nHits, nHitsJoint);
            
            
            contextualQueryTemp.add(tempUpdated);
        }
        
        // write training data
        //writeDataJointContextualAppropriateness(contextualQueryTemp);
        writeDataJointContextualAppropriatenessGeneral(contextualQueryTemp, contextTrainingFilePath);
        //writeDataRandomQueriesOnly(contextualQueryTemp, contextTrainingFilePath);

    }
    
    public void getDataJointContextualAppropriatenessGeneral_onManualVariants (int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
//        int kQueries = 100;     // #queries
//        int kTerms = 5;         // #terns in each query
//        int nHits = 1000; //indexReader.maxDoc();     // #top docs for each hits
//        int nHitsJoint = 1000;   // #top docs to be selected from joint/merged list of hits' to select 'posDocs' and/or 'negDocs'
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_RandomKTerms_avgRankBased_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocs_FixedKTerms_cust2OverkNN_timesBased_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocsOnManualVariants_timesBased_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        
        List<ContextualQuery>   contextualQueryTemp = new ArrayList<>();
        
        //while ((line = br.readLine()) != null) {    // For each joint context
        for (TRECQuery query : queries) {   // for each query (~jointContext)            
            
            // Gets +ve docs and -ve docs
            //ContextualQuery tempUpdated = getDocsByRandomQuerying(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeighted(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral_customized(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral_customized1(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getDocsByRandomQueryingWeightedGeneral_customized2(temp, kQueries, kTerms, nHits, nHitsJoint);
            //ContextualQuery tempUpdated = getRandomQueriesOnly(query, temp, kQueries, kTerms, nHits, nHitsJoint);
            ContextualQuery tempUpdated = getDocs_customized2_onManualVariants(query, nHits, nHitsJoint);
            
            
            contextualQueryTemp.add(tempUpdated);
        }
        
        // write training data
        //writeDataJointContextualAppropriateness(contextualQueryTemp);
        writeDataJointContextualAppropriatenessGeneral(contextualQueryTemp, contextTrainingFilePath);
        //writeDataRandomQueriesOnly(contextualQueryTemp, contextTrainingFilePath);

    }
    
    public void getDataContextualAppropriateness_DirectJointContext (String posTagsFilePath, String negTagsFilePath, String jointContextFilePath) throws Exception {
        
        float jointContextAppropriatenessCutOff = 0.2f;
        float jointContextAppropriatenessCutOffPos = 0.8f;
        float jointContextAppropriatenessCutOffNeg = 0.2f;
        int kQueries = 100;     // #queries
        int kTerms = 3;         // #terns in each query
        int nHits = 50; //indexReader.maxDoc();     // #top docs for each hits
        int nHitsJoint = 200;   // #top docs to be selected from joint/merged list of hits' to select 'posDocs' and/or 'negDocs'
        List<ContextualQuery>   contextualQueryTemp = new ArrayList<>();
        
        File file = new File(jointContextFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(posTagsFilePath);
        FileReader fr2 = new FileReader(file2);
        BufferedReader br2 = new BufferedReader(fr2);
        
        File file3 = new File(negTagsFilePath);
        FileReader fr3 = new FileReader(file3);
        BufferedReader br3 = new BufferedReader(fr3);
        
        String line;

        while ((line = br.readLine()) != null) {    // For each joint context
            String lineSplit[] = line.split("\t");
            String jointContext = lineSplit[0] + "-AND-" + lineSplit[1] + "-AND-" + lineSplit[2];
            System.out.print("Getting data for: " + jointContext);
//            List<TermList> grourpTermList = getAppropriateTermList(lineSplit[0]);
//            Collections.sort(grourpTermList, new cmpTermListTerm());
//            List<TermList> tripTypeTermList = getAppropriateTermList(lineSplit[1]);
//            Collections.sort(tripTypeTermList, new cmpTermListTerm());
//            List<TermList> tripDurationTermList = getAppropriateTermList(lineSplit[2]);
//            Collections.sort(tripDurationTermList, new cmpTermListTerm());
//            
//            List<TermList> jointTermlist = new ArrayList<>();
//            int nJointTermList = grourpTermList.size();
//            for (int i = 0; i < nJointTermList; ++i) {
//                TermList temp = new TermList();
//                temp.term = grourpTermList.get(i).term;
//                temp.weight = (grourpTermList.get(i).weight + tripTypeTermList.get(i).weight + tripDurationTermList.get(i).weight) / 3;
//                
//                jointTermlist.add(temp);
//            }
//            Collections.sort(jointTermlist, new cmpTermListWeight());
            
            ContextualQuery temp = new ContextualQuery();
            temp.context = jointContext;
//            String posTerms = "", negTerms = "";
//            for (int i = 0; i < jointTermlist.size(); ++i) {
//                if(jointTermlist.get(i).weight > jointContextAppropriatenessCutOffPos)
//                    posTerms += jointTermlist.get(i).term + " ";
//                else if(jointTermlist.get(i).weight < jointContextAppropriatenessCutOffNeg)
//                    negTerms += jointTermlist.get(i).term + " ";
//            }
//            temp.posTags = posTerms.split("\\s");
//            temp.negTags = negTerms.split("\\s");
            
            temp.posTags = br2.readLine().split("\\s");
            temp.negTags = br3.readLine().split("\\s");
            System.out.println("\tposTags: " + temp.posTags.length + "\tnegTags: " + temp.negTags.length);
            
            // Gets +ve docs and -ve docs
            ContextualQuery tempUpdated = getDocsByRandomQuerying(temp, kQueries, kTerms, nHits, nHitsJoint);
            
            contextualQueryTemp.add(tempUpdated);
        }
        
        // write training data
        writeDataJointContextualAppropriatenessOLD(contextualQueryTemp);

        br.close(); fr.close(); br2.close(); fr2.close(); br3.close(); fr3.close();
    }
        
    // Gets joint contextual appropriateness data
    public void getDataJointContextualAppropriateness (String jointContextFilePath, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        String contextTrainingFilePath = "/store/Data/TREC_CS/PRFDocs_FixedKTermsOld_avgRankBased_TRECCS_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        
        float jointContextAppropriatenessCutOff = 0.2f;
        float jointContextAppropriatenessCutOffPos = 0.8f;
        float jointContextAppropriatenessCutOffNeg = 0.2f;
        float initialCutOff = -1.0f;
        //int kQueries = 100;     // #queries
        //int kTerms = 5;         // #terns in each query
        //int nHits = 50; //indexReader.maxDoc();     // #top docs for each hits
        //int nHitsJoint = 200;   // #top docs to be selected from joint/merged list of hits' to select 'posDocs' and/or 'negDocs'
        List<ContextualQuery>   contextualQueryTemp = new ArrayList<>();
        
        File file = new File(jointContextFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;

        while ((line = br.readLine()) != null) {    // For each joint context
            String lineSplit[] = line.split("\t");
            String jointContext = lineSplit[0] + "-AND-" + lineSplit[1] + "-AND-" + lineSplit[2];
            String city = lineSplit[3];
            //System.out.print("Getting data for: " + jointContext);
            List<TermList> grourpTermList = getAppropriateTermList(lineSplit[0]);
            Collections.sort(grourpTermList, new cmpTermListTerm());
            List<TermList> tripTypeTermList = getAppropriateTermList(lineSplit[1]);
            Collections.sort(tripTypeTermList, new cmpTermListTerm());
            List<TermList> tripDurationTermList = getAppropriateTermList(lineSplit[2]);
            Collections.sort(tripDurationTermList, new cmpTermListTerm());
            
            List<TermList> jointTermlist = new ArrayList<>();
            List<TermList> jointTermlistWeighted = new ArrayList<>();
            int nJointTermList = grourpTermList.size();
            for (int i = 0; i < nJointTermList; ++i) {
                TermList temp = new TermList();
                temp.term = grourpTermList.get(i).term;
                temp.weight = (grourpTermList.get(i).weight + tripTypeTermList.get(i).weight + tripDurationTermList.get(i).weight) / 3;
                if(temp.weight >= initialCutOff) {  // Initial filtering. May not be needed.
                    jointTermlistWeighted.add(temp);
                }
                jointTermlist.add(temp);
            }
            Collections.sort(jointTermlist, new cmpTermListWeight());
            Collections.sort(jointTermlistWeighted, new cmpTermListWeight());
            jointTermlistWeighted = getNormalizedTermList(jointTermlistWeighted);
            
//        System.out.println();
//        int n = jointTermlistWeighted.size();
//        for (int i = 0; i < n; ++i) {
//            //sum += jointTermlistWeighted.get(i).weight;
//            System.out.println(i + ": " + jointTermlistWeighted.get(i).term + "\t\t\t" + jointTermlistWeighted.get(i).weight + "\tMAX: " + jointTermlistWeighted.get(i).max + "\tMIN: " + jointTermlistWeighted.get(i).min);
//        }
//        System.out.println("Try 1: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 2: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 3: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 4: " + getBiasedRandomIndex(jointTermlistWeighted));
//        System.out.println("Try 5: " + getBiasedRandomIndex(jointTermlistWeighted));
////        System.out.println("\nSUM: " + sum);
////        System.out.println("MIN: " + jointTermlistWeighted.get(n-1).weight);
////        System.out.println("MAX: " + jointTermlistWeighted.get(0).weight);
//        System.exit(1);

            
// 'jointTermlist' toiri. ete sob tags ache.
//            System.out.println("\n" + jointContext);
//            int n1=0, n2=0, n3=0, n4=0, n5=0, p1=0, p2=0, p3=0, p4=0, p5=0;
//            for (int i = 0; i < jointTermlist.size(); ++i) {
//                //System.out.println(jointTermlist.get(i).term + "\t" + jointTermlist.get(i).weight);
//                if(jointTermlist.get(i).weight >= 0.8f)
//                    p5++;
//                if(jointTermlist.get(i).weight >= 0.6f && jointTermlist.get(i).weight < 0.8f)
//                    p4++;
//                if(jointTermlist.get(i).weight >= 0.4f && jointTermlist.get(i).weight < 0.6f)
//                    p3++;
//                if(jointTermlist.get(i).weight >= 0.2f && jointTermlist.get(i).weight < 0.4f)
//                    p2++;
//                if(jointTermlist.get(i).weight >= 0.0f && jointTermlist.get(i).weight < 0.2f)
//                    p1++;
//                
//                if(jointTermlist.get(i).weight >= -0.2f && jointTermlist.get(i).weight < 0.0f)
//                    n1++;
//                if(jointTermlist.get(i).weight >= -0.4f && jointTermlist.get(i).weight < -0.2f)
//                    n2++;
//                if(jointTermlist.get(i).weight >= -0.6f && jointTermlist.get(i).weight < -0.4f)
//                    n3++;
//                if(jointTermlist.get(i).weight >= -0.8f && jointTermlist.get(i).weight < -0.6f)
//                    n4++;
//                if(jointTermlist.get(i).weight < -0.8f)
//                    n5++;
//            }
//            System.out.println("(-1.0 -0.8)\t(-0.8 -0.6)\t(-0.6 -0.4)\t(-0.4 -0.2)\t(-0.2 0.0)\t(0.0 0.2)\t(0.2 0.4)\t(0.4 0.6)\t(0.6 0.8)\t(0.8 1.0)");
//            //System.out.println(n5 + "\t" + n4 + "\t" + n3 + "\t" + n2 + "\t" + n1 + "\t" + p1 + "\t" + p2 + "\t" + p3 + "\t" + p4 + "\t" + p5);
//            System.out.println(n5 + "\t\t" + n4 + "\t\t" + n3 + "\t\t" + n2 + "\t\t" + n1 + "\t\t" + p1 + "\t\t" + p2 + "\t\t" + p3 + "\t\t" + p4 + "\t\t" + p5);
            //System.exit(1);
            
            
            ContextualQuery temp = new ContextualQuery();
            //temp.context = jointContext;
            temp.context = jointContext + "-AND-City:-" + city;
            temp.city = city;
            System.out.println("Getting data for: " + temp.context);
            String posTerms = "", negTerms = "";
            int nJointTermlist = jointTermlist.size();
            for (int i = 0; i < nJointTermlist; ++i) {
                if(jointTermlist.get(i).weight > jointContextAppropriatenessCutOffPos)
                    posTerms += jointTermlist.get(i).term + " ";
                else if(jointTermlist.get(i).weight < jointContextAppropriatenessCutOffNeg)
                    negTerms += jointTermlist.get(i).term + " ";
            }
            temp.posTags = posTerms.split("\\s");
            temp.negTags = negTerms.split("\\s");
            temp.posTagsWeighted = jointTermlistWeighted;
            System.out.println("\tposTags: " + temp.posTags.length + "\tnegTags: " + temp.negTags.length);
            
            // Gets +ve docs and -ve docs
            //ContextualQuery tempUpdated = getDocsByRandomQuerying(temp, kQueries, kTerms, nHits, nHitsJoint);
            ContextualQuery tempUpdated = getDocsByRandomQueryingWeighted(temp, kQueries, kTerms, nHits, nHitsJoint);
            
            contextualQueryTemp.add(tempUpdated);
        }
        
        // write training data
        writeDataJointContextualAppropriateness(contextualQueryTemp, contextTrainingFilePath);

        br.close(); fr.close();
    }
    
    // Gets raw contextual appropriateness data from Mohammad's "contextual_features.csv" and then gets joint contextual appropriateness data
    public void getDataContextualAppropriateness (String contextFilePath, String jointContextFilePath, int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        contextualAppropriateness = new ArrayList<>();
        File file = new File(contextFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;

        while ((line = br.readLine()) != null) {
            ContextualAppropriateness temp = new ContextualAppropriateness();
            String lineSplit[] = line.split(",");
            temp.context = lineSplit[3].replace(" ", "-");
            temp.category = lineSplit[2].replace("(", "").replace(")", "").replace(" ", "-").replace("-/-", " ").replace("-&-", " ");
            temp.score = Float.parseFloat(lineSplit[1]);
            temp.nAssessors = Integer.parseInt(lineSplit[0]);

            contextualAppropriateness.add(temp);
        }
        br.close(); fr.close();
        
        // Gets joint contextual appropriateness data
        getDataJointContextualAppropriateness(jointContextFilePath, kQueries, kTerms, nHits, nHitsJoint);
    }
    
    public void getqRels (String qrelFilePath) throws Exception {
        
        qrels = new ArrayList<>();
        File file = new File(qrelFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;

        while ((line = br.readLine()) != null) {
            String[] temp = line.split(" ");
            UserPreference tempQrel = new UserPreference();
            tempQrel.queryNo = Integer.parseInt(temp[0]);
            int nDocs = (temp.length-1)/2;
            tempQrel.docId = new String[nDocs];
            tempQrel.rating = new int[nDocs];
            tempQrel.nPreference = nDocs;
             int j = 1;
            for (int i = 0; i < nDocs; ++i) {
                tempQrel.docId[i] = temp[j++];
                tempQrel.rating[i] = Integer.parseInt(temp[j++]);
            }
            qrels.add(tempQrel);
        }
    }
    
    public ScoreDoc[] reRankUsingPOILevelContextualAppropriateness_TESTING_on_prefHistory(TRECQuery query) throws Exception {
        
        UserPreference uPref = userPref.get(getUPrefIndex(userPref, Integer.parseInt(query.qid)));  // All user history docs for 'qID' query/user
        trecQueryparser.getAnalyzedQuery(query, 1);
        ScoreDoc[] hitsRaw = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
        if (hitsRaw == null || hitsRaw.length == 0) //hits = retrieve(query);
            hitsRaw = retrieveCustomized(query, numHits);

        ScoreDoc[] hits = hitsRaw.clone();
        String context, classifiedPosNeg = "", classifiedPosNegGroup = "", classifiedPosNegTripType = "", classifiedPosNegTripDuration = "";
        float confidence = 0.0f, confidenceGroup = 0.0f, confidenceTripType = 0.0f, confidenceTripDuration = 0.0f;
        Classified classified, classifiedGroup, classifiedTripType, classifiedTripDuration;
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        //System.out.println(treccsQueryJson.get(treccsQueryJsonIndex).qID + ": " + "Group-type:-" +treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "\t" + "Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "\t" + "Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-"));

        for (int i = 0; i < hits.length; ++i) {
            //System.out.println(hitsRaw[i].doc + "\t" + hits[i].doc);
            System.out.println(hits[i].doc);
        }
        System.out.println("------------------------------");
        System.exit(1);
        
        hits = normalizeMinMax_hits(hits);
        //hits = normalizeEquiDist_hits(hits);
        
        //float boost = getHitsBoost(hits);
        //System.out.println(treccsQueryJson.get(treccsQueryJsonIndex).qID + ": boost = " + boost);
        
        // Gets 'confidenceMin' and 'confidenceMax' of contextual appropriateness for docs in 'hits'. Needed for Min-Max normalization.
        float confidenceMin, confidenceMax;
        context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        classified = predictPOIlevelContextualRelevance(hits[0].doc, context);
        if(classified.confidence == 0.0f)
            classified.confidence = Float.MIN_VALUE;
        if("negative".equals(classified.classifiedPosNeg)) {
            classified.confidence *= -1.0f;
        }
        confidenceMin = classified.confidence;
        confidenceMax = classified.confidence;
            
        for (int i = 1; i < hits.length; ++i) {
            context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
            classified = predictPOIlevelContextualRelevance(hits[i].doc, context);
            if(classified.confidence == 0.0f)
                classified.confidence = Float.MIN_VALUE;
            if("negative".equals(classified.classifiedPosNeg)) {
                classified.confidence *= -1.0f;
            }
            if(classified.confidence < confidenceMin)
                confidenceMin = classified.confidence;
            if(classified.confidence > confidenceMax)
                confidenceMax = classified.confidence;
        }
        
        //System.out.println("confidenceMin: " + confidenceMin + "\tconfidenceMax: " + confidenceMax);
        
        int predRel = 0, trueRel = 0, trueRelFlag = 0, trueRelAndPredRelFlag = 0, j = 0;
        for (int i = 0; i < hits.length; ++i) {
            
            String docID = indexSearcher.doc(hits[i].doc).get("docid");
            if (checkUPrefDocHistory(uPref, Integer.parseInt(query.qid), docID) != -99) {
                trueRel++;
                trueRelFlag = 1;
            }
            
            //context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-");
            context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
            classified = predictPOIlevelContextualRelevance(hits[i].doc, context);
            if(classified.confidence == 0.0f)
                classified.confidence = Float.MIN_VALUE;
            
            classified.confidence = (classified.confidence - confidenceMin) / (confidenceMax - confidenceMin);
            
            if("positive".equals(classified.classifiedPosNeg)) {
                predRel++;
                if (trueRelFlag == 1)
                    trueRelAndPredRelFlag++;

                // Swap
                if(j != i) {
                    //System.out.println(query.qid + ": Swap j = " + j + ", i = " + i + "\ti.e. " + hits[j].doc + " - " + hits[i].doc);
                    //System.out.println(query.qid + ": Swap " + hits[j].doc + " - " + hits[i].doc);
                    ScoreDoc temp = new ScoreDoc(hits[i].doc, hits[i].score);
                    hits[i] = hits[j];
                    hits[j] = temp;
                
//                    int tempDocId = hits[i].doc;
//                    hits[i].doc = hits[j].doc;
//                    hits[j].doc = tempDocId;
                }
                j++;

                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (+ve)\tUpdated score: " + (hits[i].score+classified.confidence));
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (+ve)\tUpdated confidence: " + ((classified.confidence - confidenceMin) / (confidenceMax - confidenceMin)));
                //hits[i].score *= classified.confidence;
                //hits[i].score += classified.confidence;
                //hits[i].score += (0.02f + classified.confidence);
                //hits[i].score += 0.02f;
            }
            else {
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (-ve)\tUpdated score: " + (hits[i].score-classified.confidence));
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (-ve)\tUpdated confidence: " + ((classified.confidence - confidenceMin) / (confidenceMax - confidenceMin)));
                //hits[i].score *= (-1.0f * classified.confidence);
                //hits[i].score -= classified.confidence;
                //hits[i].score -= (0.02f + classified.confidence);
                //hits[i].score -= 0.02f;
                ;
            }
            trueRelFlag = 0;
        }
        //System.out.println("-------------------------------------------------------");
        
        //System.out.println(query.qid + ": " + context + "\tAccuracy: " + predRel + " of " + hits.length + " (" + (float)predRel/hits.length*100 + "%)");
        System.out.println(query.qid + ": " + context + "\tT: " + trueRel + "\tP: " + predRel + "\tT&P: " + trueRelAndPredRelFlag + "\tT&P/T: " + trueRelAndPredRelFlag + "/" + trueRel + " (" + (float)trueRelAndPredRelFlag/trueRel*100 + "%)" + "\tT&P/P: " + trueRelAndPredRelFlag + "/" + predRel + " (" + (float)trueRelAndPredRelFlag/predRel*100 + "%)");
        
        //Arrays.sort(hits, new cmpScoreDoc());
//        for (int i = 0; i < hits.length; ++i) {
//            System.out.println(query.qid + ": " + hitsRaw[i].doc + "\t" + hits[i].doc);
//        }
//        System.out.println("-------------------------------------------------------");
        //System.exit(1);
        //return hits;
        return normalizeEquiDist_hits(hits);

        // Returns 'hitsReranked' which is top 'numHits' from 'hits', in case 'hits' has more (initially retrieved) docs
//        int nRetrieved = Math.min(hits.length, numHits);
//        ScoreDoc[] hitsReranked = new ScoreDoc[nRetrieved];
//        for (int i = 0; i < nRetrieved; ++i) {
//            hitsReranked[i] = hits[i];
//        }
//        return hitsReranked;
    }
    
    public ScoreDoc[] reRankUsingPOILevelContextualAppropriateness_TESTING(TRECQuery query, ScoreDoc[] hitsRaw) throws Exception {

        ScoreDoc[] hits = hitsRaw.clone();
        String context, classifiedPosNeg = "", classifiedPosNegGroup = "", classifiedPosNegTripType = "", classifiedPosNegTripDuration = "";
        float confidence = 0.0f, confidenceGroup = 0.0f, confidenceTripType = 0.0f, confidenceTripDuration = 0.0f;
        Classified classified, classifiedGroup, classifiedTripType, classifiedTripDuration;
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        //System.out.println(treccsQueryJson.get(treccsQueryJsonIndex).qID + ": " + "Group-type:-" +treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "\t" + "Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "\t" + "Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-"));

//        for (int i = 0; i < hits.length; ++i) {
//            System.out.println(hitsRaw[i].doc + "\t" + hits[i].doc);
//        }
//        System.out.println("------------------------------");
        
        hits = normalizeMinMax_hits(hits);
        //hits = normalizeEquiDist_hits(hits);
        
        //float boost = getHitsBoost(hits);
        //System.out.println(treccsQueryJson.get(treccsQueryJsonIndex).qID + ": boost = " + boost);
        
        // Gets 'confidenceMin' and 'confidenceMax' of contextual appropriateness for docs in 'hits'. Needed for Min-Max normalization.
        float confidenceMin, confidenceMax;
        context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        classified = predictPOIlevelContextualRelevance(hits[0].doc, context);
        if(classified.confidence == 0.0f)
            classified.confidence = Float.MIN_VALUE;
        if("negative".equals(classified.classifiedPosNeg)) {
            classified.confidence *= -1.0f;
        }
        confidenceMin = classified.confidence;
        confidenceMax = classified.confidence;
            
        for (int i = 1; i < hits.length; ++i) {
            context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
            classified = predictPOIlevelContextualRelevance(hits[i].doc, context);
            if(classified.confidence == 0.0f)
                classified.confidence = Float.MIN_VALUE;
            if("negative".equals(classified.classifiedPosNeg)) {
                classified.confidence *= -1.0f;
            }
            if(classified.confidence < confidenceMin)
                confidenceMin = classified.confidence;
            if(classified.confidence > confidenceMax)
                confidenceMax = classified.confidence;
        }
        
        //System.out.println("confidenceMin: " + confidenceMin + "\tconfidenceMax: " + confidenceMax);
        
        int predRel = 0, trueRel = 0, trueRelFlag = 0, trueRelAndPredRelFlag = 0, j = 0;
        for (int i = 0; i < hits.length; ++i) {
            
            String docID = indexSearcher.doc(hits[i].doc).get("docid");
            if (checkUPrefDoc(Integer.parseInt(query.qid), docID) != -99) {
                trueRel++;
                trueRelFlag = 1;
            }
            
            //context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-");
            context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
            classified = predictPOIlevelContextualRelevance(hits[i].doc, context);
            if(classified.confidence == 0.0f)
                classified.confidence = Float.MIN_VALUE;
            
            classified.confidence = (classified.confidence - confidenceMin) / (confidenceMax - confidenceMin);
            
            if("positive".equals(classified.classifiedPosNeg)) {
                predRel++;
                if (trueRelFlag == 1)
                    trueRelAndPredRelFlag++;

                // Swap
                if(j != i) {
                    //System.out.println(query.qid + ": Swap j = " + j + ", i = " + i + "\ti.e. " + hits[j].doc + " - " + hits[i].doc);
                    //System.out.println(query.qid + ": Swap " + hits[j].doc + " - " + hits[i].doc);
                    ScoreDoc temp = new ScoreDoc(hits[i].doc, hits[i].score);
                    hits[i] = hits[j];
                    hits[j] = temp;
                
//                    int tempDocId = hits[i].doc;
//                    hits[i].doc = hits[j].doc;
//                    hits[j].doc = tempDocId;
                }
                j++;

                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (+ve)\tUpdated score: " + (hits[i].score+classified.confidence));
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (+ve)\tUpdated confidence: " + ((classified.confidence - confidenceMin) / (confidenceMax - confidenceMin)));
                //hits[i].score *= classified.confidence;
                //hits[i].score += classified.confidence;
                //hits[i].score += (0.02f + classified.confidence);
                //hits[i].score += 0.02f;
            }
            else {
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (-ve)\tUpdated score: " + (hits[i].score-classified.confidence));
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (-ve)\tUpdated confidence: " + ((classified.confidence - confidenceMin) / (confidenceMax - confidenceMin)));
                //hits[i].score *= (-1.0f * classified.confidence);
                //hits[i].score -= classified.confidence;
                //hits[i].score -= (0.02f + classified.confidence);
                //hits[i].score -= 0.02f;
                ;
            }
            trueRelFlag = 0;
        }
        //System.out.println("-------------------------------------------------------");
        
        //System.out.println(query.qid + ": " + context + "\tAccuracy: " + predRel + " of " + hits.length + " (" + (float)predRel/hits.length*100 + "%)");
        System.out.println(query.qid + ": " + context + "\tT: " + trueRel + "\tP: " + predRel + "\tT&P: " + trueRelAndPredRelFlag + "\tT&P/T: " + trueRelAndPredRelFlag + "/" + trueRel + " (" + (float)trueRelAndPredRelFlag/trueRel*100 + "%)" + "\tT&P/P: " + trueRelAndPredRelFlag + "/" + predRel + " (" + (float)trueRelAndPredRelFlag/predRel*100 + "%)");
        
        //Arrays.sort(hits, new cmpScoreDoc());
//        for (int i = 0; i < hits.length; ++i) {
//            System.out.println(query.qid + ": " + hitsRaw[i].doc + "\t" + hits[i].doc);
//        }
//        System.out.println("-------------------------------------------------------");
        //System.exit(1);
        //return hits;
        return normalizeEquiDist_hits(hits);

        // Returns 'hitsReranked' which is top 'numHits' from 'hits', in case 'hits' has more (initially retrieved) docs
//        int nRetrieved = Math.min(hits.length, numHits);
//        ScoreDoc[] hitsReranked = new ScoreDoc[nRetrieved];
//        for (int i = 0; i < nRetrieved; ++i) {
//            hitsReranked[i] = hits[i];
//        }
//        return hitsReranked;
    }
    
    public void trainContextualRelevance_TESTING (String contextualRelevanceTrainingFilePath, int nDocsTrainPos, int nDocsTestPos, int nDocs, int trainingStatus) throws Exception {
        File file = new File(contextualRelevanceTrainingFilePath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        Random rand = new Random();
        
        System.out.println("Training = " + nDocsTrainPos + ", Testing = " + nDocsTestPos + "\n");
                
        // Read data
        String contexts[] = br.readLine().split(" ");
        int i = 0;
        while ((line = br.readLine()) != null) {
            ContextualQuery temp = new ContextualQuery();
            temp.context = contexts[i];
            temp.posTags = line.split(" ");
            temp.negTags = br.readLine().split(" ");
            temp.posDocs = br.readLine().split(" ");
            temp.negDocs = br.readLine().split(" ");
            if(trainingStatus == 0) {
                temp.bayes = new BayesClassifier<String, String>();
                contextualQuery.add(temp);
            }
            else {
                contextualQuery.get(i).posDocs = temp.posDocs;
                contextualQuery.get(i).negDocs = temp.negDocs;
            }
            i++;
        }
        br.close(); fr.close();
        
        if(trainingStatus != 0) {


        //System.out.print("Training POI level contextual relevance ");
        int nContextualQuery = contextualQuery.size();
        for (i = 0; i < nContextualQuery; ++i) { // For each context (e.g. "Trip-duration:-Night-out") train a Naive Bayes classifier
            
            String context = contextualQuery.get(i).context;
            
            // Training...
            String[] trainListPos = new String[nDocsTrainPos];
            String[] trainListNeg = new String[nDocsTrainPos];
            int k = 0, trainPosCount = 0, trainNegCount = 0;
            for (int j = 0; j < nDocsTrainPos; ++j) {
                List<TermList> topTermsPos = new ArrayList<>();
                List<TermList> topTermsNeg = new ArrayList<>();
                int randIndex = rand.nextInt(nDocs);
                
                // Training 'positive' class
                int luceneDocId = getLuceneDocId(contextualQuery.get(i).posDocs[randIndex]);
                if (luceneDocId >= 0) {
                    topTermsPos.addAll(getTermsOnly(luceneDocId));
                    String terms = "";
                    String[] positiveText = null;
                    int nTopTermsPos = topTermsPos.size();
                    for (int l = 0; l < nTopTermsPos; ++l) {
                        terms += topTermsPos.get(l).term + " ";
                    }
                    positiveText = terms.split("\\s");
                    contextualQuery.get(i).bayes.learn("positive", Arrays.asList(positiveText));
                    trainPosCount++;
                }
                trainListPos[k] = contextualQuery.get(i).posDocs[randIndex];
                
                // Training 'negative' class
                luceneDocId = getLuceneDocId(contextualQuery.get(i).negDocs[randIndex]);
                if (luceneDocId >= 0) {
                    topTermsNeg.addAll(getTermsOnly(luceneDocId));
                    String terms = "";
                    String[] negativeText = null;
                    int nTopTermsNeg = topTermsNeg.size();
                    for (int l = 0; l < nTopTermsNeg; ++l) {
                        terms += topTermsNeg.get(l).term + " ";
                    }
                    negativeText = terms.split("\\s");
                    contextualQuery.get(i).bayes.learn("negative", Arrays.asList(negativeText));
                    trainNegCount++;
                }
                trainListNeg[k] = contextualQuery.get(i).negDocs[randIndex];
                k++;
            }
            Arrays.sort(trainListPos);
            Arrays.sort(trainListNeg);
            
            // Testing...
            // Testing 'positive' class
            k = 0; int j = 0; int countPos = 0;
            while(k < nDocsTestPos && j < contextualQuery.get(i).posDocs.length) {
                String testDocID = contextualQuery.get(i).posDocs[j++];
                if(Arrays.binarySearch(trainListPos, testDocID) < 0) {
                    // Testing classifier...
                    Classified classified = predictPOIlevelContextualRelevance(getLuceneDocId(testDocID), context);
//                    if(classified.confidence == 0.0f)
//                        classified.confidence = Float.MIN_VALUE;
//                    if("negative".equals(classified.classifiedPosNeg)) {
//                        classified.confidence *= -1.0f;
//                    }
                    if("positive".equals(classified.classifiedPosNeg)) {
                        countPos++;
                    }
                    k++;
                }
            }
            
            // Testing 'negative' class
            k = 0; j = 0; int countNeg = 0;
            while(k < nDocsTestPos && j < contextualQuery.get(i).negDocs.length) {
                String testDocID = contextualQuery.get(i).negDocs[j++];
                if(Arrays.binarySearch(trainListNeg, testDocID) < 0) {
                    // Testing classifier...
                    Classified classified = predictPOIlevelContextualRelevance(getLuceneDocId(testDocID), context);
//                    if(classified.confidence == 0.0f)
//                        classified.confidence = Float.MIN_VALUE;
//                    if("negative".equals(classified.classifiedPosNeg)) {
//                        classified.confidence *= -1.0f;
//                    }
                    if("negative".equals(classified.classifiedPosNeg)) {
                        countNeg++;
                    }
                    k++;
                }
            }

            //System.out.println(i + ": " + context + "\tTrain(+ve): " + trainPosCount + "\tTrain(-ve): " + trainNegCount + "\tAccuracy: " + countPos + " of " + nDocsTestPos + " (" + (float) countPos/nDocsTestPos*100 + "%)");
            //System.out.println(i + ": " + context + "\tTrain(+ve): " + trainPosCount + "\tTrain(-ve): " + trainNegCount + "\tAccuracy (+ve): " + countPos + " of " + nDocsTestPos + " (" + (float) countPos/nDocsTestPos*100 + "%)\tAccuracy (+ve): " + countNeg + " of " + nDocsTestPos + " (" + (float) countNeg/nDocsTestPos*100 + "%)\tAccuracy(Total): " + (countPos+countNeg) + " of " + (nDocsTestPos+nDocsTestPos) + " (" + (float) (countPos+countNeg)/(nDocsTestPos+nDocsTestPos)*100 + "%)");
            System.out.println(i + ": " + context + "\tAccuracy +ve: " + countPos + " of " + nDocsTestPos + " (" + (float) countPos/nDocsTestPos*100 + "%)\t-ve: " + countNeg + " of " + nDocsTestPos + " (" + (float) countNeg/nDocsTestPos*100 + "%)\tTotal: " + (countPos+countNeg) + " of " + (nDocsTestPos+nDocsTestPos) + " (" + (float) (countPos+countNeg)/(nDocsTestPos+nDocsTestPos)*100 + "%)");
            
        }
        
        }
    }
    
    public void exploreContextualQuery (int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        // Generate training data (union) for contextual relevance and store it in file "/store/Data/TREC_CS/contextualRelevanceTraining.txt"
        //getDataContextualRelevance();

        // Generate training data (joint learning) for contextual relevance and store it in file "/store/Data/TREC_CS/contextualRelevanceTraining.txt"
//        String contextFilePath = "/store/Data/TREC_CS/contextual_features_Edited_TailRemoved.csv";
//        String posTagsFilePath = "/store/Data/TREC_CS/contextualApproPosTermsForJointContext_MohammadRaw";
//        String negTagsFilePath = "/store/Data/TREC_CS/contextualApproNegTermsForJointContext_MohammadRaw";
//        //String jointContextFilePath = "/store/Data/TREC_CS/jointContexts.txt";
//        String jointContextFilePath = "/store/Data/TREC_CS/jointContexts_city.txt";
//        getDataContextualAppropriateness(contextFilePath, jointContextFilePath, kQueries, kTerms, nHits, nHitsJoint);
//        //getDataContextualAppropriateness_DirectJointContext(posTagsFilePath, negTagsFilePath, jointContextFilePath);
//        System.exit(1);
        
        String qrelFilePath = "/store/Data/TREC_CS/queryWiseRelDocs.txt";
        getqRels(qrelFilePath);
//        for (int i = 0; i < qrels.size(); ++i) {
//            System.out.print(qrels.get(i).queryNo);
//            int nDocs = qrels.get(i).nPreference;
//            for (int j = 0; j < nDocs; ++j) {
//                System.out.print(" " + qrels.get(i).docId[j] + " " + qrels.get(i).rating[j]);
//            }
//            System.out.println();
//        }
//        System.exit(1);
        
//        contextualQuery = new ArrayList<>();
//        String contextTrainingFilePathTest = "/store/Data/TREC_CS/contextualRelevanceTraining_HeadTail_0.8_0.2.txt"; // contextualRelevanceTraining_PosNeg_0.8_0.2.txt
//        //String contextTrainingFilePathTest = "/store/Data/TREC_CS/contextualRelevanceTraining_PosNeg_0.8_0.2.txt";
//        int nDocsTrainPos = 160, nDocsTestPos = 40, nDocs = 200;
//        trainContextualRelevance_TESTING(contextTrainingFilePathTest, nDocsTrainPos, nDocsTestPos, nDocs, 0);
//        
//        contextTrainingFilePathTest = "/store/Data/TREC_CS/contextualRelevanceTraining_PosNeg_0.8_0.2.txt";
//        trainContextualRelevance_TESTING(contextTrainingFilePathTest, nDocsTrainPos, nDocsTestPos, nDocs, 1);
//        System.exit(1);
        
        // Read training data. For each context (e.g. "Trip-duration:-Night-out") train a  binary ("positive" and "negative") Naive Bayes classifier
        contextualQuery = new ArrayList<>();
        //String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining_PosNeg_0.8_0.2.txt";
        //String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining_HeadTail_0.8_0.2.txt"; // contextualRelevanceTraining_PosNeg.txt, contextualRelevanceTraining_HeadTail.txt
        //String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining_ScoreBased_HeadTail_0.8_0.2.txt";
        //String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining61_ScoreBased_PosNeg_0.8_0.2.txt";
        String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining61_weighted.txt";
        //String contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining61_weighted_unconstrained.txt";
        //String contextTrainingFilePath = "/store/Data/TREC_CS/PRFDocs_FixedKTermsOld_avgRankBased_TRECCS_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        int nDocsTrainPos = 200, nDocsTrainNeg = 200; // #docs (+ve/-ve) for training (joint cotext learning)
        trainContextualRelevance(contextTrainingFilePath, nDocsTrainPos, nDocsTrainNeg, 0);
        
        contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining_PosNeg_0.8_0.2.txt"; // contextualRelevanceTraining_PosNeg.txt, contextualRelevanceTraining_HeadTail.txt
        //contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining_ScoreBased_PosNeg_0.8_0.2.txt";
        nDocsTrainPos = 200; nDocsTrainNeg = 200; // #docs (+ve/-ve) for training (joint cotext learning)
        //trainContextualRelevance(contextTrainingFilePath, nDocsTrainPos, nDocsTrainNeg, 1);
        
//        contextTrainingFilePath = "/store/Data/TREC_CS/contextualRelevanceTraining.txt"; // contextualRelevanceTraining_PosNeg.txt, contextualRelevanceTraining_HeadTail.txt
//        nDocsTrainPos = 200; nDocsTrainNeg = 200; // #docs (+ve/-ve) for training (joint cotext learning)
//        trainContextualRelevance(contextTrainingFilePath, nDocsTrainPos, nDocsTrainNeg, 1);
        

        
        
//        int nContextualAppropriateness = contextualAppropriateness.size();
//        for (int i = 0; i < nContextualAppropriateness; ++i) {
//            System.out.println(contextualAppropriateness.get(i).context + "\t" + contextualAppropriateness.get(i).category + "\t" + contextualAppropriateness.get(i).score);
//        }
//        System.out.println("nContextualAppropriateness: " + nContextualAppropriateness);


        
//        int treccsQueryJsonIndex = getTreccsQueryJsonIndex("700");
//        System.out.println(treccsQueryJson.get(treccsQueryJsonIndex).qID + ": " + "Group-type:-" +treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "\t" + "Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "\t" + "Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-"));
        
        //int contextualQueryIndex = getContextualQueryIndex(treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-"));
        
        
//        int nTreccsQueryJson = treccsQueryJson.size();
//        for (int i = 0; i < nTreccsQueryJson; ++i) {
//            //System.out.println(treccsQueryJson.get(i).qID + ": " + treccsQueryJson.get(i).group + "\t" + treccsQueryJson.get(i).trip_type + "\t" + treccsQueryJson.get(i).duration);
//            System.out.println(treccsQueryJson.get(i).qID + ": " + "Group-type:-" +treccsQueryJson.get(i).group.replace(" ", "-") + "\t" + "Trip-type:-" + treccsQueryJson.get(i).trip_type.replace(" ", "-") + "\t" + "Trip-duration:-" + treccsQueryJson.get(i).duration.replace(" ", "-"));
//        }
        

        
     }
    
    public void exploreContextualQueryGeneral (int kQueries, int kTerms, int nHits, int nHitsJoint) throws Exception {
        
        // Generate contextualAppro data (posDocs) for each query (~jointContext)
//        getDataJointContextualAppropriatenessGeneral(kQueries, kTerms, nHits, nHitsJoint);
//        //getDataJointContextualAppropriatenessGeneral_neighbouringTerms(kQueries, kTerms, nHits, nHitsJoint);
//        //getDataJointContextualAppropriatenessGeneral_onManualVariants(kQueries, kTerms, nHits, nHitsJoint);
//        System.exit(1);
        
        // Experiments...
        contextualQuery = new ArrayList<>();
        
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_NeighbouringTerms="+numFeedbackTermsGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/RandomQueriesOnly_TRECRb_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocs_FixedKTerms_Old1_avgRankBased_TRECRb_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocsRandomQueryCust2_TREC7_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/PRFDocs_FixedKTermsOldANI_avgRankBased_TREC8_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt"; // this one
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/OldBestTREC7.txt";
        String contextTrainingFilePath = "/store/Data/TREC_CS/PRFDocs_FixedKTermsOld_avgRankBasedANIFRLM_TRECCS_kQueries="+kQueries+"_kTerms="+kTerms+"_nHits="+nHits+"_nHitsJoint="+nHitsJoint+"_RLM_Docs="+numFeedbackDocsGlobal+"_Terms="+numFeedbackTermsGlobal+"_QMIX="+QMIXGlobal+".txt";
        
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_kQueries=100_kTerms=5_nHits=50_nHitsJoints=200.txt";
        //String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextualRelevanceTrainingGeneral_kQueries=100_kTerms=5_nHits=1000_nHitsJoints=1000.txt";
        int nDocsTrainPos = 1000, nDocsTrainNeg = 1000; // #docs (+ve/-ve) for training (joint cotext learning)
        trainContextualRelevance(contextTrainingFilePath, nDocsTrainPos, nDocsTrainNeg, 0);
        //readRandomQueriesOnly(contextTrainingFilePath, nDocsTrainPos, nDocsTrainNeg, 0);
        
        
    }
    
    // Returns a 'TRECQuery' 'qNeg' with negative tags as 'qtitle'. Other info such as qid, qcity etc are same as 'query'
    public TRECQuery getQueryNegativeTags (TRECQuery query) throws Exception {
        TRECQuery qNeg = new TRECQuery();
        qNeg.qid = query.qid;
        qNeg.qtitle = query.qtitle;
        qNeg.luceneQuery = query.luceneQuery;
        qNeg.fieldToSearch = query.fieldToSearch;
        qNeg.qcity = query.qcity;
        qNeg.qlat = query.qlat;
        qNeg.qlng = query.qlng;
        qNeg.qClass = query.qClass;
        qNeg.qClassWeight = query.qClassWeight;
        
        String stringTags = "";
        for (int i = 0; i < userPrefNegativeTags.size(); ++i) {
            if(Integer.parseInt(query.qid) == userPrefNegativeTags.get(i).queryNo) {
                for (int j = 0; j < userPrefNegativeTags.get(i).nPreference; ++j) {
                    stringTags += userPrefNegativeTags.get(i).docId[j] + " ";
                }
                qNeg.qtitle = stringTags;
                return qNeg;
            }
        }
        return null;
    }

    public void loadTagsClustersWeight (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        TagsClustersWeight = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            TagsClustersWeight.add(line.split(" "));
            //Arrays.sort(TagsClustersWeight.get(i));
        }
        br.close();
        Collections.sort(TagsClustersWeight, new cmpW2VModel());
    }

    
    public void loadTagsClusters (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        TagsClusters = new ArrayList<String []>();
        int i = 0;
        while ((line = br.readLine()) != null) {
            TagsClusters.add(i, line.split(" "));
            Arrays.sort(TagsClusters.get(i));
            i++;
        }
        br.close();
//        for (i = 0; i < TagsClusters.size(); ++i) {
//            for (int j = 0; j < TagsClusters.get(i).length; ++j) {
//                System.out.print(TagsClusters.get(i)[j] + " ");
//            }
//            System.out.println("");
//        }
        //System.out.println(Arrays.binarySearch(TagsClusters.get(0), "archeology"));
        //System.exit(1);

    }
    
    // Load doc wise set of paragraph/sentence(context for BERT) vectors
    public void loadBERTDoc (String path) throws Exception {
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        BERTdoc = new ArrayList<>();
        
        line = br.readLine();
        String content[] = line.split(" ");
        String ID = content[0];
        BERTDoc tempBERT = new BERTDoc();
        tempBERT.docID = ID;
        tempBERT.vectors = new ArrayList<>();
        float[] vector = new float[768];
        for (int i = 1; i < content.length; ++i) {
            vector[i-1] = Float.parseFloat(content[i]);
        }
        tempBERT.vectors.add(vector);
        
        while ((line = br.readLine()) != null) {
            String content1[] = line.split(" ");
            if(content1[0].equals(ID)) {
                vector = new float[768];
                for (int i = 1; i < content.length; ++i) {
                    vector[i-1] = Float.parseFloat(content[i]);
                }
                tempBERT.vectors.add(vector);
            }
            else {
                BERTdoc.add(tempBERT);
                ID = content1[0];
                tempBERT = new BERTDoc();
                tempBERT.docID = ID;
                tempBERT.vectors = new ArrayList<>();
                vector = new float[768];
                for (int i = 1; i < content.length; ++i) {
                    vector[i-1] = Float.parseFloat(content[i]);
                }
                tempBERT.vectors.add(vector);
            }
        }
        BERTdoc.add(tempBERT);
        br.close(); fr.close();
        Collections.sort(BERTdoc, new cmpBERTDoc());
    }
    
    // Efficient using 'Word2vec' class
    public void loadW2V (String path) throws Exception { // Load Word2vec model

        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        W2V = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String content[] = line.split(" ");
            Word2vec temp = new Word2vec();
            temp.term = content[0];
            int j = 0;
            for (int i = 1; i < content.length; ++i) {
                temp.vector[j++] = Float.parseFloat(content[i]);
            }
            W2V.add(temp);
        }
        br.close(); fr.close();
        Collections.sort(W2V, new cmpW2V());        
    }
    
    // Similar to loadW2V() above with additional query ID for query wise context embedding (BERT)
    // File needs to be query wise sorted
    public void loadBERT (String path) throws Exception {

        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        
        bert = new ArrayList<>();
        String line;
        line = br.readLine();
        String content[] = line.split(" ");
        BERT temp = new BERT();
        
        String qID = content[0];
        String term = content[1];
        temp.qID = qID;
        temp.bertVec = new ArrayList<>();
        BERTVec tempBERTVec = new BERTVec();
        tempBERTVec.term = term;
        for (int i = 2; i < content.length; ++i) {
            tempBERTVec.vector[i - 2] = Float.parseFloat(content[i]);
        }
        temp.bertVec.add(tempBERTVec);
        
        while ((line = br.readLine()) != null) {
            content = line.split(" ");
            
            if(qID.equals(content[0])) {
                term = content[1];
                tempBERTVec = new BERTVec();
                tempBERTVec.term = term;
                for (int i = 2; i < content.length; ++i) {
                    tempBERTVec.vector[i - 2] = Float.parseFloat(content[i]);
                }
                temp.bertVec.add(tempBERTVec);
            }
            else {
                Collections.sort(temp.bertVec, new cmpBERTVec());
                bert.add(temp);
                temp = new BERT();
                
                qID = content[0];
                term = content[1];
                temp.qID = qID;
                temp.bertVec = new ArrayList<>();
                tempBERTVec = new BERTVec();
                tempBERTVec.term = term;
                for (int i = 2; i < content.length; ++i) {
                    tempBERTVec.vector[i - 2] = Float.parseFloat(content[i]);
                }
                temp.bertVec.add(tempBERTVec);
            }
        }
        Collections.sort(temp.bertVec, new cmpBERTVec());
        bert.add(temp);

        br.close(); fr.close();
    }
    
    public void loadW2Vmodel (String path) throws Exception { // Load Word2vec model

        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        //String[] keyTermVector = new String[201];
        //List<String []> W2Vmodel = new ArrayList<String []>();
        W2Vmodel = new ArrayList<String []>();
        int i = 0;
        while ((line = br.readLine()) != null) {
            W2Vmodel.add(line.split(" "));
            //W2Vmodel.add(i, line.split(" "));
            //System.out.println("Yo Man!\n-----------------------------------------\n" + i + "\t" + line);
        //System.exit(1);
            i++;
        }
        //System.exit(1);
        br.close(); fr.close();
        Collections.sort(W2Vmodel, new cmpW2VModel());
        
//        keyTermVector[0] = "beer";
//        System.out.println(Collections.binarySearch(W2Vmodel, keyTermVector, new cmpW2VModel()));
//
//        System.out.println(W2Vmodel.size() + "\t" + W2Vmodel.get(0).length);
//        for (i = 0; i < W2Vmodel.size(); ++i) {
//            for (int j = 0; j < W2Vmodel.get(i).length; ++j) {
//                System.out.print(W2Vmodel.get(i)[j] + " ");
//            }
//            System.out.println("");
//        }
//          System.exit(1);

        
//        int nDoc = indexReader.numDocs();
//        int nTopic = queries.size();
//        System.out.println("nTopic: " + nTopic);
//        for (int i = 0; i < nTopic; ++i) {
//            System.out.println("Query " + i + ": " + trecQueryparser.getAnalyzedQuery(queries.get(i), 1));
//        }
//        String content = indexSearcher.doc(0).getField("full-content").stringValue();
//        String[] terms = content.split(" ");
//        System.out.println("Content: " + terms.length);
//        String content1 = trecQueryparser.getAnalyzedQuery(queries.get(0), 1).toString();
//        String[] terms1 = content1.split(" ");
//        System.out.println("Content: " + terms1[1].replace("full-content:", ""));

        //System.out.println("Query: " + queries.get(0).qtitle);
//        System.out.println("Query: " + trecQueryparser.getAnalyzedQuery(queries.get(0), 1));
//        System.out.println("nDoc: " + nDoc);
//        System.out.println("ID: " + indexSearcher.doc(0).getField("docid").stringValue());
//        System.out.println("Content: " + indexSearcher.doc(0).getField("full-content").stringValue());
//        System.exit(1);
//        File file = new File("/store/TCD/TREC_CS/Wor2vec/trunk/64K_BOW_analysed");
//        file.createNewFile();
//        FileWriter writer = new FileWriter(file, true);
//        
//        for (int i = 0; i < nDoc; ++i) {
//            //Document doc = indexReader.document(i);
//            Document doc = indexSearcher.doc(i);
//            //System.out.println("Doc " + i + ": " + doc.getField("qQID").stringValue());
//            System.out.println("Doc: " + i + " Done!");
//            writer.write(doc.getField("full-content").stringValue() + " ");
//            writer.flush();
//        }
//        writer.close();
//        System.exit(1);
    }
    
    public int getTreccsQueryJsonIndex(String qID)  throws Exception {    // Get index of query of 'TRECId' in 'treccsQueryJson'
        TRECCSQuery temp = new TRECCSQuery();
        temp.qID = qID;
        return Collections.binarySearch(treccsQueryJson, temp, new cmpTRECCSQuery());
    }
    
    public int getFourSquareIndex(String TRECId)  throws Exception {    // Get index of Foursquare entry of 'TRECId' POI in 'foursquareData'
        FourSquareData temp = new FourSquareData();
        temp.TRECId = TRECId;
        return Collections.binarySearch(foursquareData, temp, new cmpFourSquareData());
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
    
    // Returns cosine similarity between two multi terms 'term1' (multiTerm e.g. American-Restaurant) and 'term2' (multiTerm e.g. Beer-Garden), based on W2V vectors
    public float getCosineSimilarityMultiTerms2 (String term1, String term2) throws Exception {            
        String[] multiTerms1 = term1.split("-");
        String[] multiTerms2 = term2.split("-");
        int multiTermsFlag1 = 0, multiTermsFlag2 = 0;
        float[] vec1 = new float[200];
        float[] vec2 = new float[200];

        for (int i = 0; i < multiTerms1.length; ++i) {
            //int index1 = getTermIndex(multiTerms1[i]);
            int index1 = getW2VTermIndex(multiTerms1[i]);
            if (index1 >= 0) {
                multiTermsFlag1++;
//                for (int j = 1; j < W2Vmodel.get(index1).length; ++j) {
//                    vec1[j - 1] += Float.parseFloat(W2Vmodel.get(index1)[j]); // Vector addition
//                }
                for (int j = 0; j < W2V.get(index1).vector.length; ++j) {
                    vec1[j] += W2V.get(index1).vector[j];
                }
            }
        }
        for (int i = 0; i < multiTerms2.length; ++i) {
            //int index2 = getTermIndex(multiTerms2[i]);
            int index2 = getW2VTermIndex(multiTerms2[i]);
            if (index2 >= 0) {
                multiTermsFlag2++;
//                for (int j = 1; j < W2Vmodel.get(index2).length; ++j) {
//                    vec2[j - 1] += Float.parseFloat(W2Vmodel.get(index2)[j]); // Vector addition
//                }
                for (int j = 0; j < W2V.get(index2).vector.length; ++j) {
                    vec2[j] += W2V.get(index2).vector[j];
                }
            }
        }
        if (multiTermsFlag1 > 0 && multiTermsFlag2 > 0) {
            return cosineSimilarity(vec1, vec2);
        }
  
        return 0.0f;
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
    
    // Returns cosine similarity between two terms 'term1' and 'term2', based on W2V vectors
    public float getCosineSimilarity (String term1, String term2) throws Exception {
        int index1 = getW2VTermIndex(term1);
        int index2 = getW2VTermIndex(term2);
        
        if(index1 >= 0 && index2 >= 0)
            return cosineSimilarity(W2V.get(index1).vector, W2V.get(index2).vector);
        
        return 0.0f;
    }
    
    // Returns cosine similarity between two terms 'term1' and 'term2', based on W2V vectors
//    public float getCosineSimilarity (String term1, String term2) throws Exception {
//        int index1 = getTermIndex(term1);
//        int index2 = getTermIndex(term2);
//        
//        if(index1 >= 0 && index2 >= 0) {
//            float[] vec1 = new float[200];
//            float[] vec2 = new float[200];
//            for (int j = 1; j < W2Vmodel.get(index1).length; ++j) {
//                vec1[j-1] = Float.parseFloat(W2Vmodel.get(index1)[j]);
//            }
//            for (int j = 1; j < W2Vmodel.get(index2).length; ++j) {
//                vec2[j-1] = Float.parseFloat(W2Vmodel.get(index2)[j]);
//            }
//            return cosineSimilarity(vec1, vec2);
//        }
//        
//        return 0.0f;
//    }

    // Returns cosine similarity between two terms 'term1' and 'term2', based on W2V vectors (l2norm normalized)
    public float getCosineSimilarityNormalized (String term1, String term2) throws Exception {
        int index1 = getTermIndex(term1);
        int index2 = getTermIndex(term2);
        
        if(index1 >= 0 && index2 >= 0) {
            float[] vec1 = new float[200];
            float[] vec2 = new float[200];
            for (int j = 1; j < W2Vmodel.get(index1).length; ++j) {
                vec1[j-1] = Float.parseFloat(W2Vmodel.get(index1)[j]);
            }
            for (int j = 1; j < W2Vmodel.get(index2).length; ++j) {
                vec2[j-1] = Float.parseFloat(W2Vmodel.get(index2)[j]);
            }
            return cosineSimilarity(normalizeVec(vec1), normalizeVec(vec2));
        }
        
        return 0.0f;
    }
    
    public float[] addVec(float[] a, float[] b) throws Exception {
        float[] c = new float[a.length];
        for (int i = 0; i < a.length; ++i) {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    // Returns l2norm (length) of vector 'a'
    public float l2norm(float[] a) throws Exception {
        float sum = 0.0f;

        for (int i = 0; i < a.length; ++i) {
            sum += Math.pow(a[i], 2);
        }
        return (float) Math.sqrt(sum);
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
    
    public List<Word2vec> topkW2V (int termIndex) throws Exception { // Returns list of terms (class Word2vec) sorted on cosine similarity scores
        List<Word2vec> W2Vlocal = new ArrayList<>();

        if(termIndex >= 0) {
            float[] vec1 = W2V.get(termIndex).vector;
            int nW2V = W2V.size();
            for (int i = 0; i < nW2V; ++i) {
                if(i != termIndex) {
                    float[] vec2 = W2V.get(i).vector;
                    Word2vec tempW2v = new Word2vec();
                    tempW2v.term = W2V.get(i).term;
                    tempW2v.vector = vec2;
                    tempW2v.consineScore = cosineSimilarity(vec1, vec2);
                    W2Vlocal.add(tempW2v);
                }
            }
            Collections.sort(W2Vlocal, new cmpW2VCosineSim());
            return W2Vlocal;
        }
        
        return W2Vlocal;
    }
    
    public List<Word2vec> topkW2Vmodel (int termIndex) throws Exception { // Returns list of terms (class Word2vec) sorted on cosine similarity scores
        List<Word2vec> W2V = new ArrayList<>();

        if(termIndex >= 0) {
            float[] vec1 = new float[200];
            for (int j = 1; j < W2Vmodel.get(termIndex).length; ++j) {
                vec1[j-1] = Float.parseFloat(W2Vmodel.get(termIndex)[j]);
            }
            int nW2V = W2Vmodel.size();
            for (int i = 0; i < nW2V; ++i) {
                if(i != termIndex) {
                    float[] vec2 = new float[200];
                    for (int j = 1; j < W2Vmodel.get(i).length; ++j) {
                        vec2[j-1] = Float.parseFloat(W2Vmodel.get(i)[j]);
                    }
                    Word2vec tempW2v = new Word2vec();
                    tempW2v.term = W2Vmodel.get(i)[0];
                    tempW2v.vector = vec2;
                    tempW2v.consineScore = cosineSimilarity(vec1, vec2);
                    W2V.add(tempW2v);
                }
            }
            Collections.sort(W2V, new cmpW2VCosineSim());
            return W2V;
        }
        
        return W2V;
    }
    
    public float cosineSimilarity (float[] a, float[] b) throws Exception { // Returns cosine similarity between two vectors
        float sum = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

        for (int i = 0; i < a.length; ++i) {
            sum += a[i] * b[i];
            sum1 += Math.pow(a[i], 2);
            sum2 += Math.pow(b[i], 2);
        }
        sum /= (Math.sqrt(sum1) * Math.sqrt(sum2));
        return sum;
    }

    
    public List<TRECQuery> generateSubQueries(TRECQuery query) throws Exception {
        
        List<TRECQuery> subQueries = new ArrayList<>();
        String[] queryNo = new String[TagsClustersWeight.get(0).length];
        queryNo[0] = query.qid;

        String[] terms = query.qtitle.split(" ");
        
        int nTagsCluster = TagsClusters.size();
        for (int i = 0; i < nTagsCluster; ++i) {
            subQueries.add(new TRECQuery());
            //subQueries.add(i, query);
            subQueries.get(i).fieldToSearch = query.fieldToSearch;
            subQueries.get(i).qid = query.qid;
            subQueries.get(i).qcity = query.qcity;
            subQueries.get(i).qlat = query.qlat;
            subQueries.get(i).qlng = query.qlng;
            subQueries.get(i).luceneQuery = query.luceneQuery;
            subQueries.get(i).qClass = "-1";
            subQueries.get(i).qClassWeight = 0.0f;
            subQueries.get(i).qtitle = "";
        }

        for (String term : terms) {
            for (int i = 0; i < nTagsCluster; ++i) {
                if(Arrays.binarySearch(TagsClusters.get(i), term) >= 0) {
                    subQueries.get(i).qtitle += term + " ";
                    subQueries.get(i).qClass = Integer.toString(i);
                }
            }
        }
        float sum = 0.0f;
        int nSubQuery = subQueries.size();
        for (int i = 0; i < nSubQuery; ++i) {
            if(!"-1".equals(subQueries.get(i).qClass)) {
                if(Collections.binarySearch(TagsClustersWeight, queryNo, new cmpW2VModel()) >= 0) {
                    subQueries.get(i).qClassWeight = Float.parseFloat(TagsClustersWeight.get(Collections.binarySearch(TagsClustersWeight, queryNo, new cmpW2VModel()))[i+1]);
                    sum += subQueries.get(i).qClassWeight;
                    //System.out.println(i + "-th sub-query weight\t" + Collections.binarySearch(TagsClustersWeight, queryNo, new cmpW2VModel()) + "-th query\t" + (i+1) + "-th entry");
                }
            }
        }

        // Weight adjustment
        for (int i = 0; i < nSubQuery; ++i) {
            if(!"-1".equals(subQueries.get(i).qClass) && sum != 0.0f) {
                subQueries.get(i).qClassWeight += ((1.0f - sum) * (subQueries.get(i).qClassWeight / sum));  // Adjusting the %age shortage for phase 1 missing tags
            }
        }
//        System.out.println("sum: " + sum);
        sum = 0.0f;
        //System.out.println("----------------------------------------------------------");
        for (int i = 0; i < nTagsCluster; ++i) {
            if(!"-1".equals(subQueries.get(i).qClass)) {
                //System.out.println(i + ": " + subQueries.get(i).qClass + " (" + subQueries.get(i).qClassWeight + ")\t" + subQueries.get(i).qtitle);
                sum += subQueries.get(i).qClassWeight; 
            }
        }
        //System.out.println("sum: " + sum);
//        System.exit(1);
        return subQueries;
    }
    
    // Returns Lucene doc ID for a given document ID 'TRECID'
    public int getLuceneDocId(String TRECID) throws Exception {

        ScoreDoc[] hits = null;
        TopDocs topDocs = null;

        TopScoreDocCollector collector = TopScoreDocCollector.create(1);    // Retrieve that only document with 'TRECID'
        
        Query docidQuery = new TermQuery(new Term("docid", TRECID));
        BooleanQuery booleanQuery = new BooleanQuery();
        booleanQuery.add(docidQuery, BooleanClause.Occur.MUST);

        indexSearcher.search(booleanQuery, collector);
        topDocs = collector.topDocs();
        hits = topDocs.scoreDocs;
        
        if(hits == null || hits.length == 0)
            return -1;
        
        return hits[0].doc;
    }
    
    // Returns IDF of 'term'
    public double getIdf(String term) throws IOException {
        long docCount = nDocGlobal;      // total number of documents in the index
        Term termInstance = new Term(fieldToSearch, term);
        long df = indexReader.docFreq(termInstance);       // DF: Returns the number of documents containing the term

        double idf;
        idf = Math.log((float)(docCount)/(float)(df+1));

        return idf;
    }

    // Returns TF of 'term' in document 'luceneDocID'
    public long getTF(String term, int luceneDocID) throws Exception {
        //Document doc = indexReader.document(docID);
        Terms termVector = indexReader.getTermVector(luceneDocID, fieldToSearch);
        TermsEnum itr = termVector.iterator();
        BytesRef termRef = null;
        
        while ((termRef = itr.next()) != null) {
            String termText = termRef.utf8ToString();
            if(term.equals(termText))
                return itr.totalTermFreq();
        }        
        return 0;

//        // From Doi (does same as above)
//        DocumentVector dv = new DocumentVector();
//        dv.field = fieldToSearch;
//        return dv.getTf(term, dv.getDocumentVector(luceneDocID, indexReader));
    }

    // Returns collection TF i.e. CF of 'term'
    public long getCF(String term) throws Exception {
        Term termInstance = new Term(fieldToSearch, term);
        return indexReader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).
    }

    // Returns normalized collection TF (CF) of 'term' i.e. the collection probability of 'term'
    public float getCFNormalized(String term) throws Exception {
        DocumentVector dv = new DocumentVector();
        Term termInstance = new Term(fieldToSearch, term);
        long termFreq = indexReader.totalTermFreq(termInstance); // CF: Returns the total number of occurrences of term across all documents (the sum of the freq() for each doc that has this term).

        return (float) termFreq / (float) dv.getVocabularySize(indexReader, fieldToSearch);
    }

    // Returns the document length of the document 'luceneDocID'
    public long getDocLen(int luceneDocID) throws Exception {
        long docLen = 0;
        if(luceneDocID >= 0) {
            Terms termVector = indexReader.getTermVector(luceneDocID, fieldToSearch);
            TermsEnum itr = termVector.iterator();
            BytesRef termRef = null;

            while ((termRef = itr.next()) != null) {
                docLen += itr.totalTermFreq();
            }
        }
        return docLen;
    }
    
    // Returns the size of the collection/corpus
    public long getCollectionSize() throws Exception {
        int nDoc = indexReader.maxDoc();

        long cs = 0;
        for (int i = 0; i < nDoc; ++i) {
            cs += getDocLen(i);
        }
        return cs;
    }
    
    public void parseGloveTerms () throws Exception {
//        String path = "/store/TCD/TREC_CS/Glove/gloveLexicon";
//        String path2 = "/store/TCD/TREC_CS/Glove/gloveLexiconParsed";
        String path = "/store/Data/TREC_CS/vocabularyRawUniq_stopRemoved.txt";
        String path2 = "/store/Data/TREC_CS/vocabularyRawUniq_stopRemoved_Parsed.txt";
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        String line;

        while ((line = br.readLine()) != null) {
            System.out.print("Term: " + line);
            String lineSplit[] = line.split("\t");
            String parsedTerm = parsedTerm(line);
            if(parsedTerm.equals("") || parsedTerm.equals("\\S+"))
                parsedTerm = "NULL";
            System.out.println("\tParsed: " + parsedTerm);
            writer.write(parsedTerm + "\n");
        }
        
        br.close(); fr.close();
        writer.close();
    }
    
    public void mergeGloveTerms () throws Exception {
//        String path = "/store/TCD/TREC_CS/Glove/glove.840B.300d_withParsedTermSorted.txt";
//        String path2 = "/store/TCD/TREC_CS/Glove/glove.840B.300d_withParsedTermSorted_merged.txt";
//        String path = "/store/Data/TREC_CS/transformers/BERT_vectors_parsed_unstemmed_sorted.txt";
//        String path2 = "/store/Data/TREC_CS/transformers/BERT_vectors_parsed_unstemmed_sorted_merged.txt";
        String path = "/store/TCD/TREC_CS/Wor2vec/GoogleNews-vectors-negative300_commonVocab_unstemmed_sorted.vec";
        String path2 = "/store/TCD/TREC_CS/Wor2vec/GoogleNews-vectors-negative300_commonVocab_unstemmed_sorted_merged.vec";

        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        String line;
        String lineSplit[];
        String term, parsedTerm, temp;
        int N = 300;    // vector dimensions 300D
        float vec[] = new float[N];
        int nVec = 1;   //  #vectors for same stem words (to compute the avg)
        
        line = br.readLine();
        lineSplit = line.split(" ");
        parsedTerm = lineSplit[0];
        term = lineSplit[1];
        temp = parsedTerm;
        int j = 0;
        for (int i = 2; i < lineSplit.length; ++i) {
            vec[j++] = Float.parseFloat(lineSplit[i]);
        }

        while ((line = br.readLine()) != null) {
            lineSplit = line.split(" ");
            parsedTerm = lineSplit[0];
            term = lineSplit[1];
            if(temp.equals(parsedTerm)) {
                j = 0;
                for (int i = 2; i < lineSplit.length; ++i) {
                    vec[j++] += Float.parseFloat(lineSplit[i]);
                    nVec++;
                }
            }
            else {
                writer.write(temp);
                for (int i = 0; i < vec.length; ++i) {
                    writer.write(" " + (vec[i] / nVec)); // avg
                }
                writer.write("\n");
                temp = parsedTerm;
                nVec = 1;
                j = 0;
                for (int i = 2; i < lineSplit.length; ++i) {
                    vec[j++] = Float.parseFloat(lineSplit[i]);
                }
            }
        }
        
        br.close(); fr.close();
        writer.close();
    }
    
    public void printVocabularyRaw() throws Exception {
        
        String path1 = "/store/TCD/TREC_CS/Wor2vec/trunk/63257_BOW";
        //String path1 = "/store/Data/TREC_CS/transformers/p";
        String path2 = "/store/Data/TREC_CS/vocabularyRaw.txt";
        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        String line;
        
        while ((line = br.readLine()) != null) {
            String[] content = line.split(" ");
            for (int i = 0; i < content.length; ++i) {
                String termCleaned = content[i].replaceAll("[^a-zA-Z0-9-]", "");
                if(termCleaned.isEmpty() == false)
                    writer.write(termCleaned + "\n");
            }
        }
        br.close(); fr.close();
        
        writer.close();
    }
    
    public void printVocabulary() throws Exception {
        
        //String path = "/store/Data/TREC_CS/vocabularyRaw";
        String path = "/store/Data/TRECAdhoc/vocabulary_TREC8";
        File file2 = new File(path);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        //IndexReader reader = DirectoryReader.open(dir);
        final Fields fields = MultiFields.getFields(indexReader);
        final Iterator<String> iterator = fields.iterator();

        while(iterator.hasNext()) {
            final String field = iterator.next();
            final Terms terms = MultiFields.getTerms(indexReader, field);
            final TermsEnum it = terms.iterator();
            BytesRef term = it.next();
            while (term != null) {
                //System.out.println(term.utf8ToString());
                //writer.write(term + "\n");
                writer.write(term.utf8ToString() + "\n");
                term = it.next();
            }
        }
        writer.close();
    }
    
    // Prints all terms of all docs to generate a BOW to train the W2V model
    public void printAllTerms(String path) throws Exception {
        
        File file = new File(path);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);
        
        int nDoc = indexReader.maxDoc();

        for (int i = 0; i < nDoc; ++i) {
            Document doc = indexSearcher.doc(i);
            String content = doc.get(fieldToSearch);
            String[] termContent = content.split("\\s+");
            
            for (String term : termContent) {
//                TRECQuery cont = new TRECQuery();
//                cont.fieldToSearch = fieldToSearch;
//                cont.qtitle = term;
//                
//                trecQueryparser.getAnalyzedQuery(cont, 1);
                //writer.write(cont.luceneQuery.toString(fieldToSearch) + " ");
                writer.write(term + " ");
//                if(charOnlyString(term))
//                    writer.write(term + "\n");
            }
            //writer.write(content + " ");
            System.out.println("Document " + i + " of " + nDoc + " DONE!");
        }
        
        writer.close();
    }
    
    public void printAllTermsFromRaw(String inputpath, String outputpath) throws Exception {
        
        File inputfile = new File(inputpath);
        FileReader fr = new FileReader(inputfile);
        BufferedReader br = new BufferedReader(fr);
        String line;

        File outputfile = new File(outputpath);
        outputfile.createNewFile();
        FileWriter writer = new FileWriter(outputfile, true);
        
        while ((line = br.readLine()) != null) {
//            String[] termContent = line.split(" ");
            
//            for (String term : termContent) {
//                TRECQuery cont = new TRECQuery();
//                cont.fieldToSearch = fieldToSearch;
//                cont.qtitle = term;
//                
//                trecQueryparser.getAnalyzedQuery(cont, 1);
//                writer.write(cont.luceneQuery.toString(fieldToSearch) + " ");
//            }
            TRECQuery cont = new TRECQuery();
            cont.fieldToSearch = fieldToSearch;
            cont.qtitle = line;
            trecQueryparser.getAnalyzedQuery(cont, 1);
            writer.write(cont.luceneQuery.toString(fieldToSearch) + " ");
            System.out.println(cont.luceneQuery.toString(fieldToSearch) + " ");
            //System.exit(1);
        }
        
        writer.close(); fr.close();
    }

//    // Exploring...
//    public void docExploreOld() throws Exception {
//        
//        List<TermList> topTerms = new ArrayList<>(); // Reusing MergedRanklists: nothing to do with ranklist. Just a list of terms with tf*idf etc.
//        
////        for (int i = 0; i < queryWiseUserPref.size(); ++i) {
////            for (int j = 0; j < queryWiseUserPref.get(i).length; ++j) {
////                System.out.print(queryWiseUserPref.get(i)[j] + " ");
////                System.out.print(queryWiseUserPrefRating.get(i)[j] + " ");
////            }
////            System.out.println();
////        }
//
//        for (int i = 0; i < queryWiseUserPref.size(); ++i) {
//            for (int j = 1; j < queryWiseUserPref.get(i).length; ++j) {
//                //System.out.print(queryWiseUserPref.get(i)[j] + " ");
//                List<TermList> tempList = new ArrayList<>();
//                int luceneDocId = getLuceneDocId(queryWiseUserPref.get(i)[j]); // indexSearcher.doc(luceneDocId).get("docid")
//                if(luceneDocId >= 0) {
//                    System.out.println("Doc: " + queryWiseUserPref.get(i)[j]);
//                    int rating = queryWiseUserPrefRating.get(i)[j];
//                    Terms termVector = indexReader.getTermVector(luceneDocId, fieldToSearch);
//                    TermsEnum itr = termVector.iterator();
//                    BytesRef termRef = null;
//
//                    while ((termRef = itr.next()) != null) {
//                        TermList temp = new TermList();
//                        temp.term = termRef.utf8ToString();
//                        long TF = itr.totalTermFreq();
//                        double IDF = getIdf(termRef.utf8ToString());
//                        temp.weight = (float) (TF * IDF * rating);
//                        tempList.add(temp);
//                    }
//                    Collections.sort(tempList, new cmpTermListWeight());
//                    int k = (int) Math.ceil(tempList.size() * 0.05);    // Taking top 10% terms of the doc 'luceneDocId'
//                    for (int l = 0; l < tempList.size() && l < k; ++l) {
//                        topTerms.add(tempList.get(l));
//                    }
//
//                }
//            }
//            List<TermList> topTermsUniq = getUniqTermList(topTerms);
//            
//            System.out.print(queryWiseUserPref.get(i)[0] + "\t");
//            String terms = "";
//            System.out.println("topTerms: " + topTerms.size() + "\ttopTermsUniq: " + topTermsUniq.size());
//            for (int l = 0; l < topTermsUniq.size(); ++l) {
//                //System.out.print(topTermsUniq.get(l).term + " ");
//                terms += topTermsUniq.get(l).term + " ";
//            }
//            System.out.println(terms);
//
//            queries.get(i).qtitle = terms;
//
//            
//            //System.out.println();
//        }
//
//    }
    
    // Get vocabulary i.e. print all terms (stemmed) of every selected doc (IDs of the selected docs are written in file 'inputpath'. SIGIR '21 work. Selected docs = Union of PRF docs for all queries)
    public void printAllTermsSelectedDocs() throws Exception {
        
        String inputpath = "/store/Data/TRECAdhoc/topdocsIds100Uniq_TREC8.txt";
        String outputpath = "/store/Data/TRECAdhoc/vocabularyPRFdocs100_TREC8.txt";
        File inputfile = new File(inputpath);
        FileReader fr = new FileReader(inputfile);
        BufferedReader br = new BufferedReader(fr);
        String line;

        File outputfile = new File(outputpath);
        outputfile.createNewFile();
        FileWriter writer = new FileWriter(outputfile, true);
        
        while ((line = br.readLine()) != null) {
            int luceneDocId = getLuceneDocId(line);
            if(luceneDocId >= 0) {
                String[] words = indexSearcher.doc(luceneDocId).get(FIELD_FULL_BOW).split("\\s+");
                for (int i = 0; i < words.length; ++i) {
                    writer.write(words[i] + "\n");
                }
            }
        }
        br.close(); fr.close();
        writer.close();
    }
    
    // Print (left/rigt/lr) context of every term in vocabulary (vocabulary created by uniq of printAllTermsSelectedDocs())
    public void printContext() throws Exception {
        
        String contextChoice = "right"; // 'left' / 'right' / 'lr'
        int kContext = 5; // #terms in context of a term (e.g. k=5 terms on the left/right/lr of a term)
        int nHits = 100; // #top docs to search for context
        
        String vocabPath = "/store/Data/TRECAdhoc/vocabularyUniqPRFdocs100_TREC8.txt";
        //String vocabPath = "/store/Data/TRECAdhoc/pp.txt";
        String contextOutput = "/store/Data/TRECAdhoc/vocabularyContext_"+contextChoice+"_"+kContext+"_TREC8.txt";
        
        File inputfile = new File(vocabPath);
        FileReader fr = new FileReader(inputfile);
        BufferedReader br = new BufferedReader(fr);
        
        File outputfile = new File(contextOutput);
        outputfile.createNewFile();
        FileWriter writer = new FileWriter(outputfile, true);
        
        String line;
        
        List<String> vocab = new ArrayList<>();
        
        while ((line = br.readLine()) != null) {
            vocab.add(line);
        }
        br.close(); fr.close();
        int nVocab = vocab.size();
        
        for (int i = 0; i < nVocab; ++i) { // for each term in 'vocab'
            
            String term = vocab.get(i);
            if(charOnlyString(term)) {
                ScoreDoc[] hits = null;
                TopDocs topDocs = null;

                TopScoreDocCollector collector = TopScoreDocCollector.create(nHits);

                Query termQuery = new TermQuery(new Term(FIELD_FULL_BOW, term));
                BooleanQuery booleanQuery = new BooleanQuery();
                booleanQuery.add(termQuery, BooleanClause.Occur.MUST);

                indexSearcher.search(booleanQuery, collector);
                topDocs = collector.topDocs();
                hits = topDocs.scoreDocs;

                //if(hits == null || hits.length == 0)
                if(hits != null && hits.length > 0) {
                    for (int j = 0; j < hits.length; ++j) {
                        String context[] = getContext(hits[j].doc, term, kContext, contextChoice).split(" contextSplitter ");
                        for (int k = 0; k < context.length; ++k) {
                            if(!context[k].equals(""))
                                writer.write(term + "\t" + context[k] + "\n");
                        }
                    }
                }
            }
        }
        writer.close();
    }
    
    // Check if 'context' exists in 'contexts' list
    public int contextExist(List<Context> contexts, String context) throws Exception {
        int nContexts = contexts.size();
        for (int i = 0; i < nContexts; ++i) {
            if(context.equals(contexts.get(i).context)) {
                return 1;
            }
        }
        return 0;
    }
    
    public List<Context> getUniqContext(List<Context> contexts) throws Exception {
        List<Context> contextsUniq = new ArrayList<>();
        int nContexts = contexts.size();
        for (int i = 0; i < nContexts; ++i) {
            if(contextExist(contextsUniq, contexts.get(i).context) == 0) {
                Context temp = new Context();
                temp.term = contexts.get(i).term;
                temp.context = contexts.get(i).context;
                temp.qID = contexts.get(i).qID;
                contextsUniq.add(temp);
            }
        }
        return contextsUniq;
    }
    
    // Print (left/rigt/lr) context of every term in vocabulary of 'nHits' top docs (per query)
    public void printContextSelected() throws Exception {
        String contextChoice = "left"; // 'left' / 'right' / 'lr'
        int kContext = 10; // #terms in context of a term (e.g. k=5 terms on the left/right/lr of a term)
        int nHits = 30; // #topDocs
        String contextTrainingFilePath = "/store/Data/TRECAdhoc/contextsQuerywise_"+contextChoice+"_"+kContext+"_TRECRb.txt";
        File file = new File(contextTrainingFilePath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);
        ScoreDoc[] hits = null;
        
        for (TRECQuery query : queries) { // for each query
            List<Context> contexts = new ArrayList<>();
            List<Context> contextsUniq = new ArrayList<>();
            hits = retrieveGeneral(query, nHits);
            
            for (int i = 0; i < hits.length; ++i) {
                int luceneDocId = hits[i].doc;
                String[] words = indexSearcher.doc(luceneDocId).get(FIELD_FULL_BOW).split("\\s+"); // Get all terms of the doc 'luceneDocId'
                for (int j = 0; j < words.length; ++j) { // for each term in doc 'luceneDocId'
                    //if(charOnlyString(words[j])) {
                        // keep printing lr contexts
                    String context = "";
                    Context temp = new Context();
                    if("left".equals(contextChoice)) {
                        //for (int k = Math.max((j - kContext), 0); k <= j; ++k) { // including the term in hand
                        for (int k = Math.max((j - kContext), 0); k < j; ++k) { // excluding the term in hand
                            if(context.equals(""))
                                context += words[k];
                            else
                                context += " " + words[k];
                        }
                        temp.term = words[j];
                        temp.context = context;
                        temp.qID = query.qid;
                        contexts.add(temp);
                    }
                    else if("right".equals(contextChoice)) {
                        for (int k = j; k <= Math.min((j + kContext), (words.length - 1)); ++k) {
                            if(context.equals(""))
                                context += words[k];
                            else
                                context += " " + words[k];
                        }
                        temp.term = words[j];
                        temp.context = context;
                        temp.qID = query.qid;
                        contexts.add(temp);
                    }
                    else {
                        for (int k = Math.max((j - kContext), 0); k <= Math.min((j + kContext), (words.length - 1)); ++k) {
                            if(context.equals(""))
                                context += words[k];
                            else
                                context += " " + words[k];
                        }
                        temp.term = words[j];
                        temp.context = context;
                        temp.qID = query.qid;
                        contexts.add(temp);
                    }
                    //}
                }
                
//                for (int j = 0; j < words.length; ++j) {
//                    System.out.print(words[j] + " ");
//                }
//                System.out.println("\n--------------------------------------------------");
//                for (int j = 0; j < contexts.size(); ++j) {
//                    System.out.println(contexts.get(j).term + "\t" + contexts.get(j).context);
//                }
//                System.exit(1);
            }
            
            Collections.sort(contexts, new cmpContextTerm());
            contextsUniq = getUniqContext(contexts);
            int nContextsUniq = contextsUniq.size();
            for (int i = 0; i < nContextsUniq; ++i) {
                writer.write(contextsUniq.get(i).qID + "\t" + contextsUniq.get(i).term + "\t" + contextsUniq.get(i).context + "\n");
            }
            
        }
        
        writer.close();
    }
    
    // Merge (take avg) term (context) vectors for every term per query created by printContextSelected()
    public void mergeContextSelected() throws Exception {
        String vecPath = "/store/Data/TRECAdhoc/contextsQuerywise_left_10_TRECRb_BERTVec.txt";
        String vecAvgPath = "/store/Data/TRECAdhoc/contextsQuerywise_left_10_TRECRb_BERTVecAvg.txt";
        File inputfile = new File(vecPath);
        FileReader fr = new FileReader(inputfile);
        BufferedReader br = new BufferedReader(fr);
        
        File file = new File(vecAvgPath);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);
        String line;
        
        line = br.readLine();
        String[] content = line.split(" ");
        String qID = content[0];
        String term = content[1];
        float[] vec = new float[768];
        for (int i = 2; i < content.length; ++i) {
            vec[i-2] = Float.parseFloat(content[i]);
        }
        String qIDCurrent = qID;
        String termCurrent = term;
        int nVec = 1;
        while ((line = br.readLine()) != null) {
            content = line.split(" ");
            qID = content[0];
            term = content[1];
            if(term.equals(termCurrent)) {
                for (int i = 2; i < content.length; ++i) {
                    vec[i-2] += Float.parseFloat(content[i]);
                }
                nVec++;
            }
            else {
                writer.write(qIDCurrent + " " + termCurrent);
                for (int i = 0; i < vec.length; ++i) {
                    vec[i] /= nVec;
                    writer.write(" " + vec[i]);
                }
                writer.write("\n");
                qIDCurrent = qID;
                termCurrent = term;
                nVec = 1;
                for (int i = 2; i < content.length; ++i) {
                    vec[i-2] += Float.parseFloat(content[i]);
                }
            }
        }
        writer.write(qIDCurrent + " " + termCurrent);
        for (int i = 0; i < vec.length; ++i) {
            vec[i] /= nVec;
            writer.write(" " + vec[i]);
        }
        writer.write("\n");
        br.close(); fr.close();
        writer.close();
    }
    
    // Get context (left/right/left+right 'k' terms) of the term 'term'. contextChoice = left/right/lr
    public String getContext(int luceneDocId, String term, int k, String contextChoice) throws Exception {
        String context = "";
        //System.out.println("Doc ID: " + indexSearcher.doc(luceneDocId).get(FIELD_ID));
        if(luceneDocId >= 0) {
            String[] words = indexSearcher.doc(luceneDocId).get(FIELD_FULL_BOW).split("\\s+");
            int termIndex = termExist(words, term);
            int nContext = 0;
            if(termIndex >= 0) {
                if(contextChoice == "left") {
                    if(nContext == 0) {
                        for (int i = Math.max((termIndex - k), 0); i <= termIndex; ++i) {
                            context += words[i] + " ";
                        }
                        nContext++;
                    }
                    else {
                        context += "contextSplitter ";
                        for (int i = Math.max((termIndex - k), 0); i <= termIndex; ++i) {
                            context += words[i] + " ";
                        }
                        nContext++;
                    }
                    termIndex = termExist_lr(words, term, termIndex+1, words.length); // Search done until 'termIndex'. Now search from termIndex+1.
                }
                else if(contextChoice == "right") {
                    if(nContext == 0) {
                        for (int i = termIndex; i <= Math.min((termIndex + k), (words.length - 1)); ++i) {
                            context += words[i] + " ";
                        }
                        nContext++;
                    }
                    else {
                        context += "contextSplitter ";
                        for (int i = termIndex; i <= Math.min((termIndex + k), (words.length - 1)); ++i) {
                            context += words[i] + " ";
                        }
                        nContext++;
                    }
                    termIndex = termExist_lr(words, term, termIndex+1, words.length); // Search done until 'termIndex'. Now search from termIndex+1.
                }
                else {
                    if(nContext == 0) {
                        for (int i = Math.max((termIndex - k), 0); i <= Math.min((termIndex + k), (words.length - 1)); ++i) {
                            context += words[i] + " ";
                        }
                        nContext++;
                    }
                    else {
                        context += "contextSplitter ";
                        for (int i = Math.max((termIndex - k), 0); i <= Math.min((termIndex + k), (words.length - 1)); ++i) {
                            context += words[i] + " ";
                        }
                        nContext++;
                    }
                    termIndex = termExist_lr(words, term, termIndex+1, words.length); // Search done until 'termIndex'. Now search from termIndex+1.
                }
            }            
        }
        return context;
    }
    
    // Returns list of terms ('TermList') from the document 'luceneDocId'. Terms only.
    public List<TermList> getTermsOnly(int luceneDocId) throws Exception {
        
        if(luceneDocId >= 0) {
            List<TermList> terms = new ArrayList<>();
            Terms termVector = indexReader.getTermVector(luceneDocId, fieldToSearch);
            TermsEnum itr = termVector.iterator();
            BytesRef termRef = null;

            while ((termRef = itr.next()) != null) {
                TermList temp = new TermList();
                temp.term = termRef.utf8ToString();
                temp.weight = 1.0;  // useless
                temp.rating = 4;    // useless
                terms.add(temp);
            }
            return terms;
        }

        return null;
    }

    // Returns list of terms ('TermList') from the document 'luceneDocId'.
    // 'rating': user rating of the document 'luceneDocId'
    // 'lambda': LM parameter
    public List<TermList> getTerms(int luceneDocId, int rating, float lambda, long collectionSize) throws Exception {
        
        if(luceneDocId >= 0) {
            List<TermList> terms = new ArrayList<>();
            long docLen = getDocLen(luceneDocId);
            Terms termVector = indexReader.getTermVector(luceneDocId, fieldToSearch);
            TermsEnum itr = termVector.iterator();
            BytesRef termRef = null;

            while ((termRef = itr.next()) != null) {
                TermList temp = new TermList();
                temp.term = termRef.utf8ToString();
                double TF = (double) itr.totalTermFreq() / docLen;   // Normalized TF
                //double CF = (double) collectionSize / getCF(temp.term);
                double CF = (double) getCF(temp.term) / collectionSize;
                double IDF = getIdf(termRef.utf8ToString());
                //temp.weight = (float) (TF * IDF);
                //temp.weight = (float) (TF * IDF * rating);
                
                //temp.weight = Math.log(1 + lambda / (1 - lambda) * TF * CF);    // LM term selection weighting
                temp.weight = (double) itr.totalTermFreq() / collectionSize;    // Mohammad's normalized freq. Here collectionSize = user pref size of that user
                temp.rating = rating;
                terms.add(temp);
            }
            return terms;
        }

        return null;
    }
        
    // Return 'qID' user's preferences' size i.e. total #terms in qID's preferences
    public long getUserPrefSize(int qID)  throws Exception {
        long uPrefSize = 0;
        int uPrefIndex = getuserPrefIndex(qID);
        for (int i = 0; i < userPref.get(uPrefIndex).nPreference; ++i) {
            int luceneDocId = getLuceneDocId(userPref.get(uPrefIndex).docId[i]);
            if (luceneDocId >= 0) {
                Terms termVector = indexReader.getTermVector(luceneDocId, fieldToSearch);
                TermsEnum itr = termVector.iterator();
                BytesRef termRef = null;

                while ((termRef = itr.next()) != null) {
                    uPrefSize++;
                }
            }
        }
        
        if(uPrefSize > 0)
            return uPrefSize;
        else
            return 100; // If there is no user pref doc available (probably query 779, 781), say we have 2 docs of size ~50. So, 100.
    }
    
    public void getDocWiseParagraphs() throws Exception {
        String path = "/store/Data/TREC_CS/candidatePOIs_inAniCorpus15362_And_userPrefAvailable24";
        String path2 = "/store/Data/TREC_CS/docWiseParagraphs_docIDs";
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        String line;

        while ((line = br.readLine()) != null) {
            int luceneDocId = getLuceneDocId(line);
            
            List<String> para = getParagraphs(luceneDocId);
            //System.out.println("ID: " + indexSearcher.doc(luceneDocId).get("docid") + "\t#Para: " + para.size());
            for (int i = 0; i < para.size(); ++i) {
                //System.out.println(line + " " + para.get(i));
                //writer.write(line + " " + para.get(i) + "\n");
                writer.write(line + "\n");
                //writer.write(para.get(i) + "\n");
            }
        }
        
        br.close(); fr.close();
        writer.close();

    }
    
    // Get a set of Strings from the doc 'luceneDocId'. Consider each string as a paragraph/sentence (context) of <=512 chars, to be fed to BERT.
    public List<String> getParagraphs(int luceneDocId) throws Exception {
        List<String> paragraphs = new ArrayList<>();
        String[] terms = indexSearcher.doc(luceneDocId).get("full-content").replaceAll("text ", "").replaceAll(" text", "").split("\\s");
        
        int count = 0, cutoff = 512;
        String term = "";
        for (int i = 0; i < terms.length; ++i) {
            count += terms[i].length();
            if(count < cutoff) {
                term += terms[i] + " ";
                count++;
            }
            else {
                paragraphs.add(term);
                term = "";
                count = 0;
                term += terms[i] + " ";
                count++;
            }
        }
        paragraphs.add(term);
        
        return paragraphs;
    }
    
    // Train positive model for Naive Bayes
    public void trainNaiveBayesPos(int qID, float lambda, int nTopTerms, long collectionSize, Classifier<String, String> bayes, List<TermList> topTermsPosOriginal) throws Exception {
        
        String[] positiveText = null;
        String terms = "";
        int flag = 0;
        int qIDIndex = getuserPrefIndex(qID);
        long uPrefSize = getUserPrefSize(qID);
        List<TermList> topTermsPos;
        
        for (int i = 0; i < userPref.get(qIDIndex).nPreference; ++i) {
            topTermsPos = new ArrayList<>();
            int luceneDocId = getLuceneDocId(userPref.get(qIDIndex).docId[i]);
            int rating = userPref.get(qIDIndex).rating[i];
            if (luceneDocId >= 0 && rating >= 3) {
                topTermsPos.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
                topTermsPosOriginal.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
                //System.out.println(qID + "Doc: " + userPref.get(qIDIndex).docId[i] + "\tRating: " + userPref.get(qIDIndex).rating[i]);
                //System.out.println("||||||||| rating: " + getTerms(luceneDocId, rating, lambda, collectionSize).get(0).rating);
            }
            if(!topTermsPos.isEmpty()) {
                terms = "";
                //topTermsPosUniq = getUniqTermList(topTermsPos);
                //Collections.sort(topTermsPosUniq, new cmpTermListWeight());
                
                //int nTopTermsUniq = topTermsPosUniq.size();
                int ntopTermsPos = topTermsPos.size();
                //int nTerms = Math.min(nTopTermsUniq, nTopTerms);
                int nTerms = Math.min(ntopTermsPos, nTopTerms);
                for (int l = 0; l < nTerms; ++l) {
                    //System.out.print(topTermsUniq.get(l).term + " (" + topTermsUniq.get(l).weight + ")\t");
                    //terms += topTermsPosUniq.get(l).term + " ";
                    terms += topTermsPos.get(l).term + " ";
                }
                //System.out.println(terms + "\n----------------------------------------------------------");
                positiveText = terms.split("\\s");
                bayes.learn("positive", Arrays.asList(positiveText));
                flag++;
                //System.out.println("Positive: " + terms);
            }
        }
        if(flag == 0) {
            positiveText = queries.get(qIDIndex).qtitle.split("\\s");
            //bayes.learn("positive", Arrays.asList(positiveText));
            //System.out.println("Positive(Q): " + queries.get(qID).qtitle);
            
            for (int i = 0; i < positiveText.length; ++i) {
                bayes.learn("positive", Arrays.asList(positiveText[i].split("\\s")));
                TermList temp = new TermList();
                temp.term = positiveText[i];
                temp.weight = 1.0/collectionSize;
                temp.rating = 4;
                //topTermsPos.add(temp);
                topTermsPosOriginal.add(temp);
            }
        }
        
        //return topTermsPos;
    }

    // Train negative model for Naive Bayes
    public void trainNaiveBayesNeg(int qID, float lambda, int nTopTerms, long collectionSize, Classifier<String, String> bayes, List<TermList> topTermsNegOriginal) throws Exception {
        
        String[] negativeText = null;
        String terms = "";
        int flag = 0;
        int qIDIndex = getuserPrefIndex(qID);
        long uPrefSize = getUserPrefSize(qID);
        List<TermList> topTermsNeg;
        
        for (int i = 0; i < userPref.get(qIDIndex).nPreference; ++i) {
            topTermsNeg = new ArrayList<>();
            int luceneDocId = getLuceneDocId(userPref.get(qIDIndex).docId[i]);
            int rating = userPref.get(qIDIndex).rating[i];
            if (luceneDocId >= 0 && rating < 3) {
                topTermsNeg.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
                topTermsNegOriginal.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
                //System.out.println(qID + "Doc: " + userPref.get(qIDIndex).docId[i] + "\tRating: " + userPref.get(qIDIndex).rating[i]);
                    //System.out.println("Doc: " + userPref.get(i).docId[j]);
                //System.out.println("||||||||| rating: " + getTerms(luceneDocId, rating, lambda, collectionSize).get(0).rating);
            }
            if(!topTermsNeg.isEmpty()) {
                terms = "";
                //topTermsNegUniq = getUniqTermList(topTermsNeg);
                //Collections.sort(topTermsPosUniq, new cmpTermListWeight());
                
                //int nTopTermsUniq = topTermsNegUniq.size();
                int ntopTermsNeg = topTermsNeg.size();
                //int nTerms = Math.min(nTopTermsUniq, nTopTerms);
                int nTerms = Math.min(ntopTermsNeg, nTopTerms);
                for (int l = 0; l < nTerms; ++l) {
                    //System.out.print(topTermsUniq.get(l).term + " (" + topTermsUniq.get(l).weight + ")\t");
                    //terms += topTermsNegUniq.get(l).term + " ";
                    terms += topTermsNeg.get(l).term + " ";
                }
                //System.out.println(terms + "\n----------------------------------------------------------");
                negativeText = terms.split("\\s");
                bayes.learn("negative", Arrays.asList(negativeText));
                flag++;
                //System.out.println("Negative: " + terms);
            }
        }
        if(flag == 0) {
            TRECQuery queryNegativeTags = getQueryNegativeTags(queries.get(qIDIndex));
            negativeText = queryNegativeTags.qtitle.split("\\s");
            //bayes.learn("negative", Arrays.asList(negativeText));
            //System.out.println("Negative(Q): " + queryNegativeTags.qtitle);
            
            for (int i = 0; i < negativeText.length; ++i) {
                bayes.learn("negative", Arrays.asList(negativeText[i].split("\\s")));
                TermList temp = new TermList();
                temp.term = negativeText[i];
                temp.weight = 1.0/collectionSize;
                temp.rating = 1;
                //topTermsNeg.add(temp);
                topTermsNegOriginal.add(temp);
            }
        }
        
        //return topTermsNeg;
    }
    
    // Gets Foursquare tags information
    public void getFoursquareTagInfo(int qID, List<TermList> FStagsPos, List<TermList> FStagsNeg) throws Exception {
        
        int qIDIndex = getuserPrefIndex(qID);
        //long uPrefSize = getUserPrefSize(qID);
        for (int i = 0; i < userPref.get(qIDIndex).nPreference; ++i) {
            int luceneDocId = getLuceneDocId(userPref.get(qIDIndex).docId[i]);
            int rating = userPref.get(qIDIndex).rating[i];
            if (luceneDocId >= 0) {
                int FSIndex = getFourSquareIndex(userPref.get(qIDIndex).docId[i]);
//                for (int j = 0; j < foursquareData.get(FSIndex).nCategories; ++j) {
//                    TermList temp = new TermList();
//                    temp.term = foursquareData.get(FSIndex).
//                }

// ||||||||||||||||||||| I N C O M P L E T E
                
                if(rating >= 3) {
                    
                }
                else {
                    
                }
                
            }
        
        }
    }

    // not using yet. may be needed for Mohammad's freq based score
    public String[] getTopTermsForNaiveBayes(int qID, float lambda, int nTopTerms, long collectionSize) throws Exception {
        List<TermList> topTermsNeg = new ArrayList<>();
        String terms = "";
        for (int i = 0; i < userPref.get(qID).nPreference; ++i) {
            int luceneDocId = getLuceneDocId(userPref.get(qID).docId[i]);
            int rating = userPref.get(qID).rating[i];
            if (luceneDocId >= 0 && rating >= 3) {
                topTermsNeg.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
                    //System.out.println("Doc: " + userPref.get(i).docId[j]);
                //System.out.println("||||||||| rating: " + getTerms(luceneDocId, rating, lambda, collectionSize).get(0).rating);
            }
        }
        if(!topTermsNeg.isEmpty()) {
            List<TermList> topTermsPosUniq = getUniqTermList(topTermsNeg);
            Collections.sort(topTermsPosUniq, new cmpTermListWeight());

            //System.out.println("topTerms: " + topTerms.size() + "\ttopTermsUniq: " + topTermsUniq.size());
            int nTopTermsUniq = topTermsPosUniq.size();
            int nTerms = Math.min(nTopTermsUniq, nTopTerms);
            for (int l = 0; l < nTerms; ++l) {
                //System.out.print(topTermsUniq.get(l).term + " (" + topTermsUniq.get(l).weight + ")\t");
                terms += topTermsPosUniq.get(l).term + " ";
            }
                //System.out.println("Query: " + userPref.get(i).queryNo + " (" + topTermsUniq.size() + ") " + terms);
            //System.out.println();

            //queries.get(qID).qtitle = terms;
            queries.get(qID).topTerms = topTermsPosUniq;
            queries.get(qID).nTopTerms = topTermsPosUniq.size();
        }
        else {
            terms = queries.get(qID).qtitle;
            String[] qTerms = terms.split("\\s");
            for (int i = 0; i < qTerms.length; ++i) {
                TermList temp = new TermList();
                temp.term = qTerms[i];
                temp.rating = 4;
                temp.weight = 1;
                
                topTermsNeg.add(temp);
            }
            queries.get(qID).topTerms = topTermsNeg;
            queries.get(qID).nTopTerms = topTermsNeg.size();
        }
        
        return terms.split("\\s");
    }
    
    public int getuserPrefIndex(int qID) throws Exception {
        int n = userPref.size();
        for (int i = 0; i < n; ++i) {
            if(qID == userPref.get(i).queryNo)
                return i;
        }
        return -1;
    }

    // Train the Naive Bayse classifier for query/user 'qID'
    public Classifier<String, String> trainNaiveBayes(int qID, List<TermList> topTermsPos, List<TermList> topTermsNeg) throws Exception {
        float lambda = 0.5f;    // for LM term selection
        int nTopTerms = 10000;
        long collectionSize = collectionSizeGlobal; //getCollectionSize();  // Mohammad used col size of user pref. not full col size.
        long uPrefSize = getUserPrefSize(qID);
        
        Classifier<String, String> bayes = new BayesClassifier<String, String>();
        trainNaiveBayesPos(qID, lambda, nTopTerms, uPrefSize, bayes, topTermsPos); // train on positive user preferences
        //System.out.println("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n");
        trainNaiveBayesNeg(qID, lambda, nTopTerms, uPrefSize, bayes, topTermsNeg); // train on negative user preferences
        //System.exit(1);
//        String[] unknownText3 = "basebal".split("\\s");
//        System.out.println("\n\n|||||||||||||||||||||||||||\n" + bayes.classify(Arrays.asList(unknownText3)).getCategory() + " " + bayes.classify(Arrays.asList(unknownText3)).getProbability());
        return bayes;
    }
    
    // Get Mohammad's review based score (Naive Bayes text classification) of POI/document 'luceneDocId'
    public float getReviewBasedScore(int luceneDocId, Classifier<String, String> bayes) throws Exception {
        float score = 0.0f;
        
        if(luceneDocId >= 0) {
            Terms termVector = indexReader.getTermVector(luceneDocId, fieldToSearch);
            TermsEnum itr = termVector.iterator();
            BytesRef termRef = null;
            String bow = "";
            while ((termRef = itr.next()) != null) {
                String termText = termRef.utf8ToString();
                bow += termText + " ";// itr.totalTermFreq();
            }
            String[] unknownText = bow.split("\\s");

            if ("positive".equals(bayes.classify(Arrays.asList(unknownText)).getCategory())) {
                return bayes.classify(Arrays.asList(unknownText)).getProbability();
            }
        }
        
        return score;
    }
    
    // Gets frequency of 'term' in 'topTerms' list
    public float getFrequency(List<TermList> topTerms, String term) throws Exception {
        float freq = 0.0f;
        
        int ntopTerms = topTerms.size();
        for (int i = 0; i < ntopTerms; ++i) {
            if(term.equals(topTerms.get(i).term)) {
                return (float) topTerms.get(i).weight;
            }
        }
        
        return freq;
    }
    
    // Get Mohammad's frequency based score of POI/document 'luceneDocId'
    // Now here frequency = LM term sel. weight. May want to change to TF
    public float getFrequencyBasedScore(TRECQuery query, int luceneDocId, List<TermList> topTermsPos, List<TermList> topTermsNeg) throws Exception {
        float score = 0.0f;
        
        if(luceneDocId >= 0) {
            Terms termVector = indexReader.getTermVector(luceneDocId, fieldToSearch);
            TermsEnum itr = termVector.iterator();
            BytesRef termRef = null;
            while ((termRef = itr.next()) != null) {
                String termText = termRef.utf8ToString();
                //System.out.print(termText + " ");
                // itr.totalTermFreq();
                float freqPos = getFrequency(topTermsPos, termText);
                float freqNeg = getFrequency(topTermsNeg, termText);
                float contextApproScore = getTermLevelContextualAppropriateness(query, termText);
                //float contextApproScore = getTermLevelContextualAppropriateness_singleContextBased(query, termText);
                
                //score += freqPos - freqNeg;
                score += (freqPos - freqNeg) * contextApproScore;
            }
            //System.out.println();
        }
        
        return score;
    }
    
    public float getFrequencyBasedScoreOnFoursquareTags(int luceneDocId, List<TermList> topTermsPos, List<TermList> topTermsNeg) throws Exception {
        float score = 0.0f;
        
        if(luceneDocId >= 0) {
            Terms termVector = indexReader.getTermVector(luceneDocId, fieldToSearch);
            TermsEnum itr = termVector.iterator();
            BytesRef termRef = null;
            while ((termRef = itr.next()) != null) {
                String termText = termRef.utf8ToString();
                //System.out.print(termText + " ");
                // itr.totalTermFreq();
                float freqPos = getFrequency(topTermsPos, termText);
                float freqNeg = getFrequency(topTermsNeg, termText);
                
                score += freqPos - freqNeg;
            }
            //System.out.println();
        }
        
        return score;
    }
    
    // Returns contextual information (group + trip_type + duration) available in 'query' as a string
    public String getContextualInfo(TRECQuery query) throws Exception {
        String contextualInfo = "";
        int treccsQueryIndex = getTreccsQueryJsonIndex(query.qid);
        String group = treccsQueryJson.get(treccsQueryIndex).group.replace(" ", "-");
        String trip_type = treccsQueryJson.get(treccsQueryIndex).trip_type.replace(" ", "-");
        String duration = treccsQueryJson.get(treccsQueryIndex).duration.replace(" ", "-");
        
        contextualInfo = group + " " + trip_type + " " + duration;

        return contextualInfo;
    }
    
    // Returns contextual appropriateness score of 'luceneDocId' POI for query 'query', based on ToIS term level contextual appropriateness
//    public float getContextApproScoreTermLevel(int luceneDocId, TRECQuery query) throws Exception {
//        float sim, min, max, avg;
//        List<TermList> topTerms = new ArrayList<>();
//        long collectionSize = getCollectionSize();
//        int rating = 3; // dummy
//        float lambda = 0.5f; // dummy
//        if (luceneDocId >= 0) {
//            topTerms.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
//                //System.out.println(qID + "Doc: " + userPref.get(qIDIndex).docId[i] + "\tRating: " + userPref.get(qIDIndex).rating[i]);
//            //System.out.println("||||||||| rating: " + getTerms(luceneDocId, rating, lambda, collectionSize).get(0).rating);
//        }
//        List<TermList> topTermsUniq = getUniqTermList(topTerms);
//        int nTopTermsUniq = topTermsUniq.size();
//        sim = getCosineSimilarityMultiTerms2(tags[i], terms[0]);
//        for (int i = 0; i < nTopTermsUniq; ++i) {
//            sim = getCosineSimilarityMultiTerms2(tags[i], terms[0]);
//        }
//    }
    
    // Returns contextual appropriateness score of 'luceneDocId' POI for query 'query'
    public float getContextualAppropriatenessScore(int luceneDocId, TRECQuery query) throws Exception {
        float score = 0.0f;
        String TRECId = indexSearcher.doc(luceneDocId).get("docid");
        int fsIndex = getFourSquareIndex(TRECId);
        int treccsQueryIndex = getTreccsQueryJsonIndex(query.qid);
        
        if(fsIndex >= 0) {
            // Current contextual information of query 'query'
            String group = "Group_type: " + treccsQueryJson.get(treccsQueryIndex).group;
            String trip_type = "Trip_type: " + treccsQueryJson.get(treccsQueryIndex).trip_type;
            String duration = "Trip_duration: " + treccsQueryJson.get(treccsQueryIndex).duration;
            
            float minScore = 100.0f;
            float maxScore = -100.0f;
            float sum = 0.0f, avgScore = 0.0f;
            int count = 0;
            float contextApproAvg;
            for (int i = 0; i < foursquareData.get(fsIndex).nCategories; ++i) {
                // For each (Foursquare) category tag associated with 'luceneDocId' POI,
                // compute the contextual appropriateness score with each context such group. Then take the avg.
                float contextAppro1 = getContextApproScore(group, foursquareData.get(fsIndex).categories.get(i));
                float contextAppro2 = getContextApproScore(trip_type, foursquareData.get(fsIndex).categories.get(i));
                float contextAppro3 = getContextApproScore(duration, foursquareData.get(fsIndex).categories.get(i));
                
                int nContextAvailable = 0;
                if(contextAppro1 != 0.0f)
                    nContextAvailable++;
                if(contextAppro2 != 0.0f)
                    nContextAvailable++;
                if(contextAppro3 != 0.0f)
                    nContextAvailable++;
                    
//                System.out.println(nContextAvailable + " of 3 Contexts available.");
                if(nContextAvailable != 0) {
                    contextApproAvg = (contextAppro1 + contextAppro2 + contextAppro3) / nContextAvailable;
                
//                    System.out.println("minScore: " + minScore + "\tcontextApproAvg: " + contextApproAvg);
                    if(contextApproAvg < minScore)
                        minScore = contextApproAvg;
                    if(contextApproAvg > maxScore)
                        maxScore = contextApproAvg;
                    sum += contextApproAvg;
                    count++;

//                    System.out.println(TRECId + ": " + group + " VS " + foursquareData.get(fsIndex).categories.get(i) + " Score: " + contextAppro1);
//                    System.out.println(TRECId + ": " + trip_type + " VS " + foursquareData.get(fsIndex).categories.get(i) + " Score: " + contextAppro2);
//                    System.out.println(TRECId + ": " + duration + " VS " + foursquareData.get(fsIndex).categories.get(i) + " Score: " + contextAppro3);
//                    System.out.println("|||||||||||| Avg.: " + contextApproAvg + "\t");
                }
            }
            if(count != 0)
                avgScore = sum/count;
            if(minScore == 100.0f)
                minScore = 0.0f;
            if(maxScore == -100.0f)
                maxScore = 0.0f;
            // Take the minimum of appropriateness scores of all Foursquare categories associated with 'luceneDocId' POI
//            System.out.println("---------------------------------------------------");
//            System.out.println(TRECId + ": minScore: " + minScore + "\tmaxScore: " + maxScore + "\tavgScore: " + avgScore);
//            System.out.println("---------------------------------------------------");
            return minScore;
            //return maxScore;
            //return avgScore;
        }
        
        return score;
    }
    
    // Returns contextual appropriateness score \in [-1, +1] for 'currentContext' and 'currentCategory' pair, based on Mohammad's crowdsource data
    public float getContextApproScore(String currentContext, String currentCategory) throws Exception {
        float score = 0.0f;
        
        for (int i = 0; i < context.length; ++i) {
            if(context[i].equals(currentContext) && category[i].equals(currentCategory))
                return contextCategoryScore[i];
                //return normalizeMinMax(contextCategoryScore[i], 1.0f, -1.0f);
        }
        
        return score;
    }
    
    // Returns the amount of boost needed on 'hits' based on avg distance between docs in 'hits'
    public float getHitsBoost(ScoreDoc[] hits) throws Exception {
        float boost = 0.0f;
        for (int i = 0; i < hits.length-1; ++i) {
            boost += (hits[i].score - hits[i+1].score);
        }
        
        return (boost / (hits.length-1));
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
    
    public ScoreDoc[] reRankFilteringOnly(TRECQuery query, ScoreDoc[] hitsModel, ScoreDoc[] hitsToRerank) throws Exception {
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        ScoreDoc[] hits = null;
        for (int i = 1; i < hitsToRerank.length; ++i) {
            if(docExist2(hitsModel, hitsToRerank[i]) >= 0) {
                hitsTemp.add(hitsToRerank[i]);
            }
        }
        
        if (hitsTemp.size() > 0) {
            hits = hitsTemp.toArray(new ScoreDoc[0]);
        }
        return hits;
    }
    
    public ScoreDoc[] reRankUsingKLDiv(TRECQuery query, ScoreDoc[] hitsModel, ScoreDoc[] hitsToRerank) throws Exception {
//        TopDocs topDocs = new TopDocs(hits.length, hits hits[0].score);
//        rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
//        HashMap<String, WordProbability> hashmap_PwGivenR = rlm.RM3(query, topDocs);
//        //HashMap<String, WordProbability> hashmap_PwGivenR = rlm.RM3_2(query, topDocs);
//
//        // Re-ranking using KL-Div
//        return convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD
        
        TopDocs topDocsModel = new TopDocs(hitsModel.length, hitsModel, hitsModel[0].score);
        rlm.setFeedbackStatsDirect(topDocsModel, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
        HashMap<String, WordProbability> hashmap_PwGivenR = rlm.RM3(query, topDocsModel);
        //HashMap<String, WordProbability> hashmap_PwGivenR = rlm.RM3_2(query, topDocs);
        //System.exit(1);

        // Re-ranking using KL-Div
        TopDocs topDocsToRerank = new TopDocs(hitsToRerank.length, hitsToRerank, hitsToRerank[0].score);
        return convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocsToRerank));  // Re-ranking using KLD

//        ScoreDoc[] hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocsToRerank));
//        Arrays.sort(hits, new cmpScoreDoc());
//        return hits;
    }
    
    public ScoreDoc[] reRankUsingPOILevelContextualAppropriateness(TRECQuery query, ScoreDoc[] hits) throws Exception {

        String context, classifiedPosNeg = "", classifiedPosNegGroup = "", classifiedPosNegTripType = "", classifiedPosNegTripDuration = "";
        float confidence = 0.0f, confidenceGroup = 0.0f, confidenceTripType = 0.0f, confidenceTripDuration = 0.0f;
        Classified classified, classifiedGroup, classifiedTripType, classifiedTripDuration;
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        //System.out.println(treccsQueryJson.get(treccsQueryJsonIndex).qID + ": " + "Group-type:-" +treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "\t" + "Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "\t" + "Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-"));

        hits = normalizeMinMax_hits(hits);
        //hits = normalizeEquiDist_hits(hits);
        
        //float boost = getHitsBoost(hits);
        //System.out.println(treccsQueryJson.get(treccsQueryJsonIndex).qID + ": boost = " + boost);
        
        
        // Gets 'confidenceMin' and 'confidenceMax' of contextual appropriateness for docs in 'hits'. Needed for Min-Max normalization.
        float confidenceMin, confidenceMax;
        //context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-City:-" + query.qcity;
        //System.out.println("ANIcontext: " + context + "\tIndex: " + getContextualQueryIndex(context));

        classified = predictPOIlevelContextualRelevance(hits[0].doc, context);
        if(classified.confidence == 0.0f)
            classified.confidence = Float.MIN_VALUE;
        if("negative".equals(classified.classifiedPosNeg)) {
            classified.confidence *= -1.0f;
        }
        confidenceMin = classified.confidence;
        confidenceMax = classified.confidence;
            
        for (int i = 1; i < hits.length; ++i) {
            //context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
            context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-City:-" + query.qcity;
            classified = predictPOIlevelContextualRelevance(hits[i].doc, context);
            if(classified.confidence == 0.0f)
                classified.confidence = Float.MIN_VALUE;
            if("negative".equals(classified.classifiedPosNeg)) {
                classified.confidence *= -1.0f;
            }
            if(classified.confidence < confidenceMin)
                confidenceMin = classified.confidence;
            if(classified.confidence > confidenceMax)
                confidenceMax = classified.confidence;
        }
        
        //System.out.println("confidenceMin: " + confidenceMin + "\tconfidenceMax: " + confidenceMax);
        
        for (int i = 0; i < hits.length; ++i) {
            
//            context = "Group-type:-" +treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-");
//            classifiedGroup = predictPOIlevelContextualRelevance(hits[i].doc, context);
//            
//            context = "Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-");
//            classifiedTripType = predictPOIlevelContextualRelevance(hits[i].doc, context);
//            
//            context = "Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
//            classifiedTripDuration = predictPOIlevelContextualRelevance(hits[i].doc, context);
            
            //context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-");
            //context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
            context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-City:-" + query.qcity;
            classified = predictPOIlevelContextualRelevance(hits[i].doc, context);
            if(classified.confidence == 0.0f)
                classified.confidence = Float.MIN_VALUE;
            
            classified.confidence = (classified.confidence - confidenceMin) / (confidenceMax - confidenceMin);
            
            if("positive".equals(classified.classifiedPosNeg)) {
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (+ve)\tUpdated score: " + (hits[i].score+classified.confidence));
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (+ve)\tUpdated confidence: " + ((classified.confidence - confidenceMin) / (confidenceMax - confidenceMin)));
                //hits[i].score *= classified.confidence;
                //hits[i].score += classified.confidence;
                //hits[i].score += (0.02f + classified.confidence);
                hits[i].score += 0.02f;
            }
            else {
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (-ve)\tUpdated score: " + (hits[i].score-classified.confidence));
                //System.out.println(i + ": docID: " + hits[i].doc + "\tRaw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + " (-ve)\tUpdated confidence: " + ((classified.confidence - confidenceMin) / (confidenceMax - confidenceMin)));
                //hits[i].score *= (-1.0f * classified.confidence);
                //hits[i].score -= classified.confidence;
                //hits[i].score -= (0.02f + classified.confidence);
                hits[i].score -= 0.02f;
            }
                

            
            //System.out.println(i + ":\tGroup: " + classifiedGroup.classifiedPosNeg + "\tTripType: " + classifiedTripType.classifiedPosNeg + "\tTripDuration: " + classifiedTripDuration.classifiedPosNeg);
            
            //System.out.println("Raw score: " +  hits[i].score + "\tConfidence score: " + classified.confidence + "\tUpdated score: " +  (hits[i].score * classified.confidence));

            
//            if("positive".equals(classifiedGroup.classifiedPosNeg) && "positive".equals(classifiedTripType.classifiedPosNeg) && "positive".equals(classifiedTripDuration.classifiedPosNeg)) {
//                //System.out.println("Raw score: " +  hits[i].score + "\tConfidence score: " + ((classifiedGroup.confidence + classifiedTripType.confidence + classifiedTripDuration.confidence)/3) + "\tUpdated score: " +  (hits[i].score * ((classifiedGroup.confidence + classifiedTripType.confidence + classifiedTripDuration.confidence)/3)));
//                
//                //hits[i].score += (classifiedGroup.confidence + classifiedTripType.confidence + classifiedTripDuration.confidence) / 3;
//                //hits[i].score *= (classifiedGroup.confidence + classifiedTripType.confidence + classifiedTripDuration.confidence) / 3;
//            }
//            else if("positive".equals(classifiedPosNegGroup) && "positive".equals(classifiedPosNegTripType)) {
//                System.out.println("Group+Type\tRaw score: " +  hits[i].score + "\tConfidence score: " + ((confidenceGroup + confidenceTripType) / 2) + "\tUpdated score: " +  (hits[i].score + ((confidenceGroup + confidenceTripType) / 2)));
//                hits[i].score += (confidenceGroup + confidenceTripType) / 2;
//            }
//            else if("positive".equals(classifiedPosNegGroup) && "positive".equals(classifiedPosNegTripDuration)) {
//                System.out.println("Group+Duration\tRaw score: " +  hits[i].score + "\tConfidence score: " + ((confidenceGroup + confidenceTripType + confidenceTripDuration)/3) + "\tUpdated score: " +  (hits[i].score + ((confidenceGroup + confidenceTripType + confidenceTripDuration)/3)));
//                hits[i].score += (confidenceGroup + confidenceTripDuration) / 2;
//            }
//            else if("positive".equals(classifiedPosNegTripType) && "positive".equals(classifiedPosNegTripDuration)) {
//                hits[i].score += (confidenceTripType + confidenceTripDuration) / 2;
//            }
//            else if("positive".equals(classifiedPosNegGroup)) {
//                hits[i].score += confidenceGroup;
//            }
//            else if("positive".equals(classifiedPosNegTripType)) {
//                hits[i].score += confidenceTripType;
//            }
//            else {
//                hits[i].score += confidenceTripDuration;
//            }

            //hits[i].score *= contextApproScore;
            //hits[i].score = contextApproScore;
        }
        
        Arrays.sort(hits, new cmpScoreDoc());
        return hits;
        
//        System.out.println("\n------------------------------------- After re-ranking -------------------------------------");
//        for (int i = 0; i < hits.length; ++i) {
//            System.out.println(i + ": docID: " + hits[i].doc + "\tScore: " +  hits[i].score);
//        }
//        System.out.println("\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
        //System.exit(1);
        
        // Returns 'hitsReranked' which is top 'numHits' from 'hits', in case 'hits' has more (initially retrieved) docs
//        int nRetrieved = Math.min(hits.length, numHits);
//        ScoreDoc[] hitsReranked = new ScoreDoc[nRetrieved];
//        for (int i = 0; i < nRetrieved; ++i) {
//            hitsReranked[i] = hits[i];
//        }
//        return hitsReranked;
    }
    
    public ScoreDoc[] reRankUsingContextualAppropriatenessSVM(TRECQuery query, ScoreDoc[] hits) throws Exception {
        
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        String context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        
        for (int i = 0; i < hits.length; ++i) {
            String TRECId = indexSearcher.doc(hits[i].doc).get("docid");
            double contextApproScore = getPoiContextualAppropriatenessSVM(TRECId, context);
            
            if(contextApproScore > 0)
                hits[i].score += contextApproScore; //Math.exp(-contextApproScore);
            else if(contextApproScore < 0)
                hits[i].score -= contextApproScore; //Math.exp(-1 * contextApproScore);
            else
                hits[i].score += 0.0f;
            
            //hits[i].score *= contextApproScore;
            //hits[i].score += contextApproScore;
            //System.out.println(query.qid + ": " + TRECId + "\t" + contextApproScore);
        }
        //System.out.println();
        
        Arrays.sort(hits, new cmpScoreDoc());
        return hits;
    }
    
    public ScoreDoc[] reRankUsingContextualAppropriateness(TRECQuery query, ScoreDoc[] hits) throws Exception {
        for (int i = 0; i < hits.length; ++i) {
            float contextApproScore = getContextualAppropriatenessScore(hits[i].doc, query);
            contextApproScore = normalizeMinMax(contextApproScore, 1.0f, -1.0f);
            
            if(contextApproScore >= 0.5f) {
                hits[i].score += contextApproScore;
            }
            //hits[i].score *= contextApproScore;
            //hits[i].score = contextApproScore;
        }
        
        Arrays.sort(hits, new cmpScoreDoc());
        return hits;
    }
    
    public ScoreDoc[] retrieveWithNaiveBayes_OverBM25(TRECQuery query, int nHits) throws Exception {
        List<TermList> topTermsPos = new ArrayList<>();
        List<TermList> topTermsNeg = new ArrayList<>();
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        ScoreDoc[] initialHits = retrieveCustomized(query, numHits);
        
        // Train 'bayes' classifier and update 'topTermsPos' and 'topTermsNeg' from user preferences
        Classifier<String, String> bayes = trainNaiveBayes(Integer.parseInt(query.qid), topTermsPos, topTermsNeg);
        
//        System.out.println("Positive model:");
//        for (int i = 0; i < topTermsPos.size(); ++i) {
//            //System.out.print(topTermsPos.get(i).term + " ");
//            System.out.print(topTermsPos.get(i).term + " (" + topTermsPos.get(i).weight + ") ");
//        }
//        System.out.println();
//        System.out.println("\nNegative model:");
//        for (int i = 0; i < topTermsNeg.size(); ++i) {
//            //System.out.print(topTermsNeg.get(i).term + " ");
//            System.out.print(topTermsNeg.get(i).term + " (" + topTermsNeg.get(i).weight + ") ");
//        }
//        System.out.println("\n|||||||||||||||||||||||||||||||||||||||||||||||");
        
        int j = 0;
        
        int nDoc = indexReader.maxDoc();
        for (int i = 0; i < initialHits.length; ++i) {
            
                int docId = initialHits[i].doc;
                float reviewBasedScore = getReviewBasedScore(docId, bayes);
                //float frequencyBasedScore = getFrequencyBasedScore(query, docId, topTermsPos, topTermsNeg);
                //float contextApproScore = getContextualAppropriatenessScore(docId, query);
                //contextApproScore = normalizeMinMax(contextApproScore, 1.0f, -1.0f);
                //frequencyBasedScore = normalizeMinMax(frequencyBasedScore, 10000, -10000);
                
                //if(contextApproScore > 0.0f) { // i.e. if it is contextually appropriate
                    float docScore = reviewBasedScore;
                    //float docScore = frequencyBasedScore;
                    //float docScore = contextApproScore;
                    //float docScore = reviewBasedScore + frequencyBasedScore;
                    //float docScore = (reviewBasedScore + frequencyBasedScore) * contextApproScore;
                    //float docScore = normalizeMinMax((reviewBasedScore + frequencyBasedScore), 100, -100) * contextApproScore;
                    //float docScore = reviewBasedScore + frequencyBasedScore + contextApproScore;
                    //float docScore = reviewBasedScore * frequencyBasedScore * contextApproScore;

                    //System.out.println("query " + query.qid + ": " + j++ + " docId: " + docId + "\t" + reviewBasedScore + "\t" + frequencyBasedScore + "\t" + contextApproScore + "\tSUM: " + docScore);
                    //System.out.println("query " + query.qid + ": " + j++ + " docId: " + docId + "\tdocScore: " + docScore);
                    //if(docScore > 0.0f) {
                        ScoreDoc temp = new ScoreDoc(docId, docScore);
                        hitsTemp.add(temp);
                    //} 
                //}
        }

        if(!hitsTemp.isEmpty()) {
            Collections.sort(hitsTemp, new cmpScoreDoc());
            int n = Math.min(nHits, hitsTemp.size());
            ScoreDoc[] hits = new ScoreDoc[n];
            for (int i = 0; i < n; ++i) {
                hits[i] = hitsTemp.get(i);
            }
            return hits;
            //return reRankUsingContextualAppropriateness(query, hits);
        }
        else
            return null;
    }
    
    public ScoreDoc[] retrieveWithNaiveBayes(TRECQuery query, int nHits) throws Exception {

        List<TermList> topTermsPos = new ArrayList<>();
        List<TermList> topTermsNeg = new ArrayList<>();
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        
        // Train 'bayes' classifier and update 'topTermsPos' and 'topTermsNeg' from user preferences
        Classifier<String, String> bayes = trainNaiveBayes(Integer.parseInt(query.qid), topTermsPos, topTermsNeg);

        List<TermList> topTermsPosUniq = getUniqTermList(topTermsPos);
        List<TermList> topTermsNegUniq = getUniqTermList(topTermsNeg);
//        System.out.println("Positive model:");
//        for (int i = 0; i < topTermsPos.size(); ++i) {
//            //System.out.print(topTermsPos.get(i).term + " ");
//            System.out.print(topTermsPos.get(i).term + " (" + topTermsPos.get(i).weight + ") ");
//        }
//        System.out.println();
//        System.out.println("\nNegative model:");
//        for (int i = 0; i < topTermsNeg.size(); ++i) {
//            //System.out.print(topTermsNeg.get(i).term + " ");
//            System.out.print(topTermsNeg.get(i).term + " (" + topTermsNeg.get(i).weight + ") ");
//        }
//        System.out.println("\n|||||||||||||||||||||||||||||||||||||||||||||||");
        
        int j = 0;
        
        int nDoc = indexReader.maxDoc();
        for (int i = 0; i < nDoc; ++i) {
            
            if(query.qcity.equals(indexSearcher.doc(i).get("cityId"))) {
                int docId = i;
                float reviewBasedScore = getReviewBasedScore(i, bayes);
                //float frequencyBasedScore = getFrequencyBasedScore(query, i, topTermsPos, topTermsNeg);
                float frequencyBasedScore = getFrequencyBasedScore(query, i, topTermsPosUniq, topTermsNegUniq);
                float contextApproScore;
                //contextApproScore = getContextualAppropriatenessScore(i, query); // OLD
                
                int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
                //String context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-");
                String context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
                
                String TRECId = indexSearcher.doc(i).get("docid");
                //contextApproScore = (float) getPoiContextualAppropriatenessSVM(TRECId, context);
            
// Open this for POI level contextual relevance                
//                Classified classified = predictPOIlevelContextualRelevance(docId, context);
//
//                if("positive".equals(classified.classifiedPosNeg)) {
//                    contextApproScore = classified.confidence;
//                    //contextApproScore = 0.02f;
//                }
//                else {
//                    contextApproScore = (-1.0f * classified.confidence);
//                    //contextApproScore = -0.02f;
//                }

                
                //contextApproScore = normalizeMinMax(contextApproScore, 1.0f, -1.0f);
                //frequencyBasedScore = normalizeMinMax(frequencyBasedScore, 10000, -10000);
                
                //if(contextApproScore > 0.0f) { // i.e. if it is contextually appropriate
                    //float docScore = reviewBasedScore;
                    //float docScore = frequencyBasedScore;
                    //float docScore = contextApproScore;
                    float docScore = reviewBasedScore + frequencyBasedScore;
                    //float docScore = reviewBasedScore + frequencyBasedScore + contextApproScore;
//                    if(contextApproScore > 0.0f)
//                        docScore += Math.exp(-contextApproScore);
//                    else if(contextApproScore < 0.0f)
//                        docScore -= Math.exp(-1 * contextApproScore);
//                    else
//                         docScore += 0.0f; // i.e. contextApproScore = 0.0f
                    
                    //float docScore = (reviewBasedScore + frequencyBasedScore) * contextApproScore;
                    //float docScore = normalizeMinMax((reviewBasedScore + frequencyBasedScore), 100, -100) * contextApproScore;
                    //float docScore = reviewBasedScore + frequencyBasedScore + contextApproScore;
                    //float docScore = reviewBasedScore * frequencyBasedScore * contextApproScore;

                    //System.out.println("query " + query.qid + ": " + j++ + " docId: " + docId + "\t" + reviewBasedScore + "\t" + frequencyBasedScore + "\t" + contextApproScore + "\tSUM: " + docScore);
                    //System.out.println("query " + query.qid + ": " + j++ + " docId: " + docId + "\tdocScore: " + docScore);
                    //if(docScore > 0.0f) {
                        ScoreDoc temp = new ScoreDoc(docId, docScore);
                        hitsTemp.add(temp);
                    //} 
                //}
                
            }
        }

        if(!hitsTemp.isEmpty()) {
            Collections.sort(hitsTemp, new cmpScoreDoc());
            int n = Math.min(nHits, hitsTemp.size());
            ScoreDoc[] hits = new ScoreDoc[n];
            for (int i = 0; i < n; ++i) {
                hits[i] = hitsTemp.get(i);
            }
            return hits;
            //return reRankUsingContextualAppropriateness(query, hits);
        }
        else
            return null;
    }
    
    public ScoreDoc[] naiveBayesExplore(TRECQuery query) throws Exception {
        
        ScoreDoc[] hits = retrieveWithNaiveBayes(query, numHits);
        //ScoreDoc[] hits = retrieveWithNaiveBayes_OverBM25(query, numHits);
        //System.out.println("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
        //System.out.println("hits Len: " + hits.length);
        //System.exit(1);
        return hits;
        
        
        
//        Classifier<String, String> bayes = new BayesClassifier<String, String>();
//        
//        String[] positiveText = "I love sunny days".split("\\s");
//        bayes.learn("positive", Arrays.asList(positiveText));
//        
//        positiveText = "I hate cactus".split("\\s");
//        bayes.learn("negative", Arrays.asList(positiveText));
//
//        String[] negativeText = "I hate rain".split("\\s");
//        bayes.learn("negative", Arrays.asList(negativeText));
//        
//        String[] unknownText1 = "today is a sunny day".split("\\s");
//        String[] unknownText2 = "there will be rain".split("\\s");
//        String[] unknownText3 = "sunny day".split("\\s");
//
//        System.out.println(bayes.classify(Arrays.asList(unknownText3)).getCategory() + " " + bayes.classify(Arrays.asList(unknownText3)).getProbability());
        
    }
    
    // Exploring...
    public void docExplore(float lambda, int nTopTerms) throws Exception {
        
//        int nUserPref = userPref.size();
//        for (int i = 0; i < nUserPref; ++i) {
//            System.out.print(userPref.get(i).queryNo + "\t");
//            for (int j = 0; j < userPref.get(i).nPreference; ++j) {
//                System.out.print(userPref.get(i).docId[j] + " " + userPref.get(i).rating[j] + " ");
//            }
//            System.out.println();
//        }

        long collectionSize = collectionSizeGlobal; //getCollectionSize();
        int nUserPref = userPref.size();
        for (int i = 0; i < nUserPref; ++i) {
            //System.out.println(userPref.get(i).queryNo + " ||||||||||||||||||||||||||||||");
            int nPref = userPref.get(i).nPreference;
            List<TermList> topTerms = new ArrayList<>();
            for (int j = 0; j < nPref; ++j) {
                //System.out.print(userPref.get(i).docId[j] + " " + userPref.get(i).rating[j] + " ");
                int luceneDocId = getLuceneDocId(userPref.get(i).docId[j]); // indexSearcher.doc(luceneDocId).get("docid")
                int rating = userPref.get(i).rating[j];
                if(luceneDocId >= 0 && rating >= 3) {
                    topTerms.addAll(getTerms(luceneDocId, rating, lambda, collectionSize));
                    //System.out.println("Doc: " + userPref.get(i).docId[j]);
                    //System.out.println("||||||||| rating: " + getTerms(luceneDocId, rating, lambda, collectionSize).get(0).rating);
                }
            }
            
            if(!topTerms.isEmpty()) {
                List<TermList> topTermsUniq = getUniqTermList(topTerms);
                Collections.sort(topTermsUniq, new cmpTermListWeight());

                String terms = "";
                //System.out.println("topTerms: " + topTerms.size() + "\ttopTermsUniq: " + topTermsUniq.size());
                int nTopTermsUniq = topTermsUniq.size();
                int nTerms = Math.min(nTopTermsUniq, nTopTerms);
                for (int l = 0; l < nTerms; ++l) {
                    //System.out.print(topTermsUniq.get(l).term + " (" + topTermsUniq.get(l).weight + ")\t");
                    terms += topTermsUniq.get(l).term + " ";
                }
                //System.out.println("Query: " + userPref.get(i).queryNo + " (" + topTermsUniq.size() + ") " + terms);
                //System.out.println();

                //queries.get(i).qtitle = terms;
                queries.get(i).topTerms = topTermsUniq;
                queries.get(i).nTopTerms = topTermsUniq.size();
  
            }
            else {
                //System.out.println("Query: " + userPref.get(i).queryNo + " (0) " + " Empty!");
                queries.get(i).topTerms = null;
                queries.get(i).nTopTerms = 0;
            }
            
            //System.exit(1);
            //System.out.println();
        }

    }

    public ScoreDoc[] CombSUM_TopTerms_And_retrieveCustomized(TRECQuery query, int numHits, int nTerms) throws Exception {
        ScoreDoc[] hits1 = null, hits2 = null, hitsMerged = null;
        List<MultipleRanklists> multiRankList = new ArrayList<>();
        MultipleRanklists tempMultiRankList;
        
        if(query.nTopTerms > 0) {
            
            tempMultiRankList = new MultipleRanklists();
            hits1 = retrieveCustomizedTopTerms(query, numHits, nTerms);
            tempMultiRankList.hits = hits1;
            tempMultiRankList.nDocs = numHits;
            tempMultiRankList.tagClass = "0";
            tempMultiRankList.weight = 0.5f;
            multiRankList.add(tempMultiRankList);

            tempMultiRankList = new MultipleRanklists();
            TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
            hits2 = retrieveCustomizedTopTerms1(queryFiltered, numHits, 100);
            //hits2 = retrieveCustomized(query, numHits);
            tempMultiRankList.hits = hits2;
            tempMultiRankList.nDocs = numHits;
            tempMultiRankList.tagClass = "1";
            tempMultiRankList.weight = 0.5f;
            multiRankList.add(tempMultiRankList);
            
            //hitsMerged = hits2;
            hitsMerged = mergeRanklists(updateAvgScoreRanklists(multiRankList));
        }
        else {
            //hitsMerged = retrieveCustomized(query, numHits);
            TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
            hitsMerged = retrieveCustomizedTopTerms1(queryFiltered, numHits, 100);
        }

        return hitsMerged;
    }
    
    public ScoreDoc[] CombSUM_TopTerms_And_retrieveCustomizedFiltered(TRECQuery query, int numHits, int nTerms) throws Exception {
        ScoreDoc[] hits1 = null, hits2 = null, hitsMerged = null;
        List<MultipleRanklists> multiRankList = new ArrayList<>();
        MultipleRanklists tempMultiRankList;
        
        if(query.nTopTerms > 0) {
            
            tempMultiRankList = new MultipleRanklists();
            //hits1 = retrieveCustomizedTopTerms(query, numHits, nTerms);
            hits1 = retrieveCustomizedTopTerms(query, numHits, 100);
            tempMultiRankList.hits = hits1;
            tempMultiRankList.nDocs = numHits;
            tempMultiRankList.tagClass = "0";
            tempMultiRankList.weight = 0.5f;
            multiRankList.add(tempMultiRankList);

            docExplore(0.5f, nTerms);
            TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
            tempMultiRankList = new MultipleRanklists();
            //hits2 = retrieveCustomized(query, numHits);
            hits2 = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);
            tempMultiRankList.hits = hits2;
            tempMultiRankList.nDocs = numHits;
            tempMultiRankList.tagClass = "1";
            tempMultiRankList.weight = 0.5f;
            multiRankList.add(tempMultiRankList);
            
            hitsMerged = mergeRanklists(updateAvgScoreRanklists(multiRankList));
        }
        else
            hitsMerged = retrieveCustomized(query, numHits);

        return hitsMerged;
    }

    public ScoreDoc[] CombSUM_KDERLM_And_ClassificatTag(TRECQuery query) throws Exception {
        ScoreDoc[] hits1 = null, hits2 = null, hitsMerged = null;
        List<MultipleRanklists> multiRankList = new ArrayList<>();
        MultipleRanklists tempMultiRankList;
        
        tempMultiRankList = new MultipleRanklists();
        hits1 = naiveBayesExplore(query);
        tempMultiRankList.hits = hits1;
        tempMultiRankList.nDocs = numHits;
        tempMultiRankList.tagClass = "0";
        tempMultiRankList.weight = 0.5f;
        multiRankList.add(tempMultiRankList);

        tempMultiRankList = new MultipleRanklists();
        hits2 = RM3Explore(query, 0, 0.8f, 0.8f);
        tempMultiRankList.hits = hits2;
        tempMultiRankList.nDocs = numHits;
        tempMultiRankList.tagClass = "1";
        tempMultiRankList.weight = 0.5f;
        multiRankList.add(tempMultiRankList);

        hitsMerged = mergeRanklists(updateAvgScoreRanklists(multiRankList));

        return hitsMerged;
    }
    
//    public float getTermWeight(List<TermList> terms, String term) throws Exception {
//        int n = terms.size();
//        for (int i = 0; i < n; ++i) {
//            if(term.equals(terms.get(i).term))
//                return (float) terms.get(i).weight;
//        }
//        
//        return 0.0f;
//    }
    
    public float getCosineSimilarityBetweenDocs_PreComputed(String qID, int luceneDocId1, int luceneDocId2) throws Exception {
        String docID1 = indexSearcher.doc(luceneDocId1).get(FIELD_ID);
        String docID2 = indexSearcher.doc(luceneDocId2).get(FIELD_ID);
        return getDocSim(qID, docID1, docID2);
    }
    
    public float getCosineSimilarityBetweenDocs(int luceneDocId1, int luceneDocId2) throws Exception {
        int rating = 4;         // garbage
        float lambda = 0.5f;    // garbage
        long collectionSize = collectionSizeGlobal; //getCollectionSize();
        List<TermList> terms1 = getTerms(luceneDocId1, rating, lambda, collectionSize);
        List<TermList> terms2 = getTerms(luceneDocId2, rating, lambda, collectionSize);
        List<TermList> terms = new ArrayList<>();
        terms.addAll(terms1);
        terms.addAll(terms2);
        
        List<TermList> uniqTerms = getUniqTermList(terms);
        int nUniqTerms = uniqTerms.size();
        float[] vector1 = new float[nUniqTerms];
        float[] vector2 = new float[nUniqTerms];
        
        for (int i = 0; i < nUniqTerms; ++i) {
            vector1[i] = getFrequency(terms1, uniqTerms.get(i).term);
            vector2[i] = getFrequency(terms2, uniqTerms.get(i).term);
        }
        
        //System.out.println("(" + luceneDocId1 + ", " + luceneDocId2 + ")\tSim: " + cosineSimilarity(vector1, vector2));
        return cosineSimilarity(vector1, vector2);        
    }
    
    public float getCosineSimilarityBetweenDocs_Smart(List<TermList>  docVec1, List<TermList>  docVec2) throws Exception {
        
        int n1 = docVec1.size();
        int n2 = docVec2.size();
        int i = 0, j = 0;
        float sum = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
        
        while (i < n1 && j < n2) 
        {   if(docVec1.get(i).term.compareTo(docVec2.get(j).term) < 0) {
                sum1 += Math.pow(docVec1.get(i).weight, 2);
                i++;
                //arr3[k++] = arr1[i++];
            }
            else if(docVec1.get(i).term.compareTo(docVec2.get(j).term) > 0) {
                sum2 += Math.pow(docVec2.get(j).weight, 2);
                j++;
                //arr3[k++] = arr2[j++]; 
            }
            else {
                //sum += a[i] * b[i];
                sum += docVec1.get(i).weight * docVec2.get(j).weight;
                //sum1 += Math.pow(a[i], 2);
                sum1 += Math.pow(docVec1.get(i).weight, 2);
                //sum2 += Math.pow(b[i], 2);
                sum2 += Math.pow(docVec2.get(j).weight, 2);
                i++;
                j++;
            }
        }
        while (i < n1) {
            sum1 += Math.pow(docVec1.get(i).weight, 2);
            i++;
        }
        while (j < n2) {
            sum2 += Math.pow(docVec2.get(j).weight, 2);
            j++;
        }
        sum /= (Math.sqrt(sum1) * Math.sqrt(sum2));
        
        return sum;        
    }
    
    public float getDocSim(String qID, String docID1, String docID2) throws Exception {
        DocumentSimilarity temp = new DocumentSimilarity();
        //temp.q_d_d = qID + "\t" + docID1 + "\t" + docID2;
        temp.q_d_d = docID1 + "\t" + docID2;
        int docSimIndex = Collections.binarySearch(docSim, temp, new cmpDocumentSimilarity());
        if(docSimIndex >= 0)
            return docSim.get(docSimIndex).score;
        else {
            System.out.println(qID + "\t" + docID1 + "\t" + docID2 + "\tNOT FOUND!");
            return 0.0f;
        }
    }
    
    public void readNxNMatrix_Smart() throws Exception {
        
        //String path = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNClusterOnUQV_TREC8.txt";
        String path = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNClusterOnSingleQuery_TREC7.txt";
        System.out.println("queryWiseNxNMatrixfor_kNNClusterFilePath set to: " + path);
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        docSim = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            DocumentSimilarity temp = new DocumentSimilarity();
            String[] content = line.split("\t");
            String docIDrow = content[0];
            String docIDcol = content[1];
            float cosineSim = Float.parseFloat(content[2]);
            temp.q_d_d = docIDrow + "\t" + docIDcol;
            temp.score = cosineSim;
            docSim.add(temp);
        }
        Collections.sort(docSim, new cmpDocumentSimilarity());
    }
    
    public void readNxNMatrix() throws Exception {
        
        String path = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNClusterOverOld_TREC8.txt";
        System.out.println("queryWiseNxNMatrixfor_kNNClusterFilePath set to: " + path);
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        docSim = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            DocumentSimilarity temp = new DocumentSimilarity();
            String[] content = line.split("\t");
            String qID = content[0];
            String docIDrow = content[1];
            String docIDcol = content[2];
            float cosineSim = Float.parseFloat(content[3]);
            temp.q_d_d = qID + "\t" + docIDrow + "\t" + docIDcol;
            temp.score = cosineSim;
            docSim.add(temp);
        }
        Collections.sort(docSim, new cmpDocumentSimilarity());
    }
    
    public void generateNxNMatrix() throws Exception {
        
        //String path = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNCluster.txt";
        String path = "/store/Data/TRECAdhoc/requiredDocDocPairs_TECRb.txt";
        File file = new File(path);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);
        
        for (TRECQuery query : queries) {
            
            ScoreDoc[] hits = retrieveGeneral(query, 100);  // N = 100 i.e. #top docs
//        trecQueryparser.getAnalyzedQuery(query, 1);
//        ScoreDoc[] hits = getPOILevelContextualApproDocsGeneral(query, 100);

            for (int i = 0; i < hits.length; ++i) {
                System.out.println(query.qid + ": Computing for doc " + i);
                String docIdRow = indexSearcher.doc(hits[i].doc).get(FIELD_ID);
                for (int j = 0; j < hits.length; ++j) {
                    String docIdCol = indexSearcher.doc(hits[j].doc).get(FIELD_ID);
                    //float cosineSim = getCosineSimilarityBetweenDocs(hits[i].doc, hits[j].doc);
                    //writer.write(query.qid + "\t" + docIdRow + "\t" + docIdCol + "\t" + cosineSim + "\n");
                    writer.write(docIdRow + "\t" + docIdCol + "\n");
                }
            }
        }
        writer.close();
    }
    
    public void PP1() throws Exception {
        String path1 = "/store/Data/TREC_CS/transformers/BERT_vectors_docWise1.txt";
        String path2 = "/store/Data/TREC_CS/transformers/BERT_vectors_docWise2.txt";
        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        String line;
        
        while ((line = br.readLine()) != null) {
            String[] content = line.split(" ");
            String ID = content[0];
            writer.write(ID);
            for (int i = content.length-768; i < content.length; ++i) {
                writer.write(" " + content[i]);
            }
            writer.write("\n");
        }
        br.close(); fr.close();
    }
    
    public void PP() throws Exception {
        String path1 = "/store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt";
        String path2 = "/store/Data/TREC_CS/qrelStats.txt";
        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        String line;
        
        line = br.readLine();
        String[] content = line.split("\t");
        String qID = content[0];
        int rel = Integer.parseInt(content[3]);
        
        String qID1;
        int relSum = 0;
        qID1 = qID;
        
        if(rel >= 1.0f)
            relSum ++;
        
        while ((line = br.readLine()) != null) {
            content = line.split("\t");
            qID = content[0];
            rel = Integer.parseInt(content[3]);
            if(qID.equals(qID1)) {
                if(rel >= 1.0f)
                    relSum ++;
            }
            else {
                writer.write(qID1 + "\t" + relSum + "\n");
                relSum = 0;
                qID1 = qID;

                if(rel >= 1.0f)
                    relSum ++;
            }
        }
        writer.write(qID1 + "\t" + relSum + "\n");
        br.close(); fr.close();
        
        writer.close();
        
    }
    
    public void writekNNDocs() throws Exception {
        
        String path = "/store/Data/TRECAdhoc/kNNDocsTRECRb.txt";

        File file2 = new File(path);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
                
        for (TRECQuery query : queries) {
            int kNN = 20;
            float threshold = 0.25f;
            float lambda = 0.4f;
            int nCluster = 20;
            ScoreDoc[] hits = getClusterBasedTopDocs(query, kNN, threshold, lambda, nCluster);
            //System.out.println("hits len: " + hits.length);
            writer.write(query.qid);
            for (int j = 0; j < 10; ++j) {
                if(j < hits.length)
                    writer.write(" " + indexSearcher.doc(hits[j].doc).get(FIELD_ID));
                else
                    writer.write(" Dummy");
            }
            writer.write("\n");
        }

        writer.close();        
    }
    
    public List<ContextualQuery> getkNNDocs(String path) throws Exception {
        //String path1 = "/store/Data/TRECAdhoc/kNNDocsTREC6.txt";
        
        List<ContextualQuery> contextualQuerykNN = new ArrayList<>();

        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
                        
        while ((line = br.readLine()) != null) {
            String[] splited = line.split(" ");
            ContextualQuery temp = new ContextualQuery();
            temp.context = splited[0];
            temp.posDocs = new String[10];
            int j = 0;
            for (int i = 1; i < splited.length; ++i) {
                temp.posDocs[j++] = splited[i];
            }
            contextualQuerykNN.add(temp);
        }
        br.close(); fr.close();
        return contextualQuerykNN;
    }
    
    public void PP2() throws Exception {
        String path1 = "/store/Data/TRECAdhoc/requiredDocIdsUniq";
        String path2 = "/store/Data/TRECAdhoc/requiredDocDocPairsUniq_TEC8.txt";
        String path3 = "/store/Data/TRECAdhoc/requiredDocDocPairsUniqReduced_TEC8.txt";
        File file1 = new File(path2);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        String line;
        
        List<String> list1 = new ArrayList<>();
        List<String> list2 = new ArrayList<>();
        
        while ((line = br.readLine()) != null) {
            String[] content = line.split("\t");
            list1.add(content[0]);
            list2.add(content[1]);
        }
        br.close(); fr.close();
        file1 = new File(path1);
        fr = new FileReader(file1);
        br = new BufferedReader(fr);
        
        File file2 = new File(path3);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        int nList = list1.size();
        int j = 1;
        while ((line = br.readLine()) != null) {
            System.out.println(j++ + ": " + line);
            for (int i = 0; i < nList; ++i) {
                if(line.equals(list1.get(i))) {
                    writer.write(line + "\t" + list2.get(i) + "\n");
                }
                else if(line.equals(list2.get(i))) {
                    writer.write(line + "\t" + list1.get(i) + "\n");
                }
            } 
        }
        writer.close();
        
    }
    
    public void generateNxNMatrixForManualQVariants() throws Exception {
        
        String path1 = "/store/Data/TRECAdhoc/rest";
        //String path2 = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNCluster.txt";
        String path2 = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNClusterOnUQV_Part3_TREC8.txt";

        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        String line;
        int i = 1;
        while ((line = br.readLine()) != null) {
            String[] content = line.split("\t");
            String docID1 = content[0];
            String docID2 = content[1];
            float cosineSim = getCosineSimilarityBetweenDocs(getLuceneDocId(docID1), getLuceneDocId(docID2));
            System.out.println(i++ + ": " + docID1 + "\t" + docID2 + "\t" + cosineSim);
            writer.write(docID1 + "\t" + docID2 + "\t" + cosineSim + "\n");
        }
        br.close(); fr.close();
        
        writer.close();
    }
    
    public List<TermList> getDocVec(List<DocumentVec> docVecs, String docID) throws Exception {
        DocumentVec temp = new DocumentVec();
        temp.docID = docID;
        
        int index = Collections.binarySearch(docVecs, temp, new cmpDocVec());
        if(index >= 0)
            return docVecs.get(index).terms;
        else
            return null;
    }
    
    public List<DocumentVec> readDocVecs(String path) throws Exception {
        
        File file1 = new File(path);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        List<DocumentVec> docVecs = new ArrayList<>();
        String line;
        System.out.println("Reading Document Vectors...");
        int i = 1;
        while ((line = br.readLine()) != null) {
            String TRECdocId = line;
            int rating = 4;         // garbage
            float lambda = 0.5f;    // garbage
            long collectionSize = collectionSizeGlobal; //getCollectionSize();
            List<TermList> terms = getTerms(getLuceneDocId(TRECdocId), rating, lambda, collectionSize);
            Collections.sort(terms, new cmpTermListTerm());
            DocumentVec temp = new DocumentVec();
            temp.docID = TRECdocId;
            temp.terms = terms;
            docVecs.add(temp);
            System.out.println("Document " + i++ + "\tDone!");
        }
        br.close(); fr.close();
        Collections.sort(docVecs, new cmpDocVec());
        
        return docVecs;
    }
    
    public void generateNxNMatrixForkNN_Smart() throws Exception {
        
        //String path0 = "/store/Data/TRECAdhoc/requiredDocIdsUniq";
        String path0 = "/store/Data/TRECAdhoc/requiredDocIdsUniq_TRECRb";
        //String path1 = "/store/Data/TRECAdhoc/restForPart3";
        String path1 = "/store/Data/TRECAdhoc/requiredDocDocPairsUniq_TECRb.txt";
        String path2 = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNCluster_TRECRb.txt";
        //String path2 = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNClusterOnUQV_Part3_TREC8.txt";
        
        List<DocumentVec> docVecs = readDocVecs(path0);
        System.out.println("----------------------FINISHED------------------------");

        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        int i = 1;
        String line;
        while ((line = br.readLine()) != null) {
            String[] content = line.split("\t");
            String docID1 = content[0];
            String docID2 = content[1];
            List<TermList> docVec1 = getDocVec(docVecs, docID1);
            List<TermList> docVec2 = getDocVec(docVecs, docID2);
            //float cosineSim = getCosineSimilarityBetweenDocs(getLuceneDocId(docID1), getLuceneDocId(docID2));
            float cosineSim_Smart = getCosineSimilarityBetweenDocs_Smart(docVec1, docVec2);
            //System.out.println(i++ + ": " + docID1 + "\t" + docID2 + "\t" + cosineSim);
            System.out.println(i++ + ": " + docID1 + "\t" + docID2 + "\t" + cosineSim_Smart);
            //System.out.println("CosineSim:\t" + docID1 + "\t" + docID2 + "\t" + cosineSim);
            //System.out.println("CosineSimSmart:\t" + docID1 + "\t" + docID2 + "\t" + cosineSim_Smart);
            //writer.write(docID1 + "\t" + docID2 + "\t" + cosineSim + "\n");
            writer.write(docID1 + "\t" + docID2 + "\t" + cosineSim_Smart + "\n");
        }
        br.close(); fr.close();
        
        writer.close();
    }
        
    public void getDocDocPairsForNxNMatrixOnUQV() throws Exception {
        
        String path = "/store/Data/TRECAdhoc/queryWiseNxNMatrixfor_kNNCluster.txt";
        //String path = "/store/Data/TRECAdhoc/requiredDocDocPairsOnUQV_TRECRb.txt";
        File file = new File(path);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);
        
        for (TRECQuery query : queries) {
            
            //int contextualQueryIndex = getContextualQueryIndex(query.qid);
            int contextualQueryIndex = getContextualQueryManualIndex(query.qid);
            
            //int nQuery = contextualQuery.get(contextualQueryIndex).randomQueries.size();
            int nQuery = contextualQueryManual.get(contextualQueryIndex).randomQueries.size();

            for (int k = 0; k < nQuery; ++k) {
                TRECQuery queryTemp = new TRECQuery();
                queryTemp.qid = query.qid;
                //queryTemp.qtitle = contextualQuery.get(contextualQueryIndex).randomQueries.get(i).qtitle;
                //queryTemp.qtitle = contextualQuery.get(contextualQueryIndex).randomQueries.get(k).qtitle;
                queryTemp.qtitle = contextualQueryManual.get(contextualQueryIndex).randomQueries.get(k).qtitle;
                queryTemp.fieldToSearch = query.fieldToSearch;
                ScoreDoc[] hits = retrieveGeneral(queryTemp, 100);
                
                if(hits != null) {
                    for (int i = 0; i < hits.length; ++i) {
                        System.out.println(query.qid + ": Computing for doc " + i);
                        String docIdRow = indexSearcher.doc(hits[i].doc).get(FIELD_ID);
                        for (int j = 0; j < hits.length; ++j) {
                            String docIdCol = indexSearcher.doc(hits[j].doc).get(FIELD_ID);
                            //float cosineSim = getCosineSimilarityBetweenDocs(hits[i].doc, hits[j].doc);
                            //writer.write(query.qid + "\t" + docIdRow + "\t" + docIdCol + "\t" + cosineSim + "\n");
                            //writer.write("dummyQId\t" + docIdRow + "\t" + docIdCol + "\t" + cosineSim + "\n");
                            writer.write(docIdRow + "\t" + docIdCol + "\n");
                        }
                    }
                }
            }
        }
        writer.close();
    }
    
    // Get 'kNN' docs from 'hits' for the document 'docID'
    public int[] getkNNDocs(TRECQuery query, int luceneDocId, ScoreDoc[] hits, int kNN, float threshold) throws Exception {
        int[] docIDs = new int[kNN];
        List<TermList> docs = new ArrayList<>();
        
        for (int i = 0; i < hits.length; ++i) {
            TermList temp = new TermList();
            temp.term = Integer.toString(hits[i].doc);
            //temp.weight = getCosineSimilarityBetweenDocs(luceneDocId, hits[i].doc);
            temp.weight = getCosineSimilarityBetweenDocs_PreComputed(query.qid, luceneDocId, hits[i].doc);
            
            docs.add(temp);
        }
        Collections.sort(docs, new cmpTermListWeight());
        
        int count = 0;
        int kNN_updated = Math.min(kNN, hits.length);
        //for (int i = 0; i < kNN; ++i) {
        for (int i = 0; i < kNN_updated; ++i) {
            if(docs.get(i).weight >= threshold)
                count++;
            docIDs[i] = Integer.parseInt(docs.get(i).term);
        }
        
        int[] docIDsFiltered = new int[count];
        for (int i = 0; i < count; ++i) {
            docIDsFiltered[i] = docIDs[i];
        }
        
        //return docIDs;
        return docIDsFiltered;
    }
    
    // Get P(w|Clu) = freq(w, Clu) / |Clu| (Lee, Croft & Allan. A cluster-based resampling method for pseudo-relevance feedback. SIGIR '08)
    public float getWGivenCluster(String w, DocumentCluster docCluster, float lambda) throws Exception {
        long clusterLen = 0;
        long collectionSize = collectionSizeGlobal; //getCollectionSize();
        long freqW_Cluster = 0;
        long freqW_Collection = getCF(w);
        
        for (int i = 0; i < docCluster.docIds.length; ++i) {
            freqW_Cluster += getTF(w, docCluster.docIds[i]);    // freq(w, Clu) = \sum{freq(w, D), D \in Clu}
            clusterLen += getDocLen(docCluster.docIds[i]);
        }

        return ((clusterLen/(float)(clusterLen+lambda)) * ((float)freqW_Cluster/clusterLen)) + ((lambda/(float)(clusterLen+lambda)) * ((float)freqW_Collection/collectionSize));
    }

    // Get P(Q|Clu) = \prod{P(q_i|Clu), q_i \in Q} i.e. the score of the cluster 'docCluster' (for clsuter ranking) based on 'query' (Lee, Croft & Allan. A cluster-based resampling method for pseudo-relevance feedback. SIGIR '08)
    public float getClusterScore(TRECQuery query, DocumentCluster docCluster, float lambda) throws Exception {
        
        String[] qTerms = query.luceneQuery.toString(fieldToSearch).split(" ");
        float QGivenCluster = 1.0f;
        for (int i = 0; i < qTerms.length; ++i) {
            QGivenCluster *= getWGivenCluster(qTerms[i], docCluster, lambda);
        }
        
        return QGivenCluster;
    }
    
    // Get top 'nHits' docs based on clusters (Lee, Croft & Allan. A cluster-based resampling method for pseudo-relevance feedback. SIGIR '08)
    public ScoreDoc[] getClusterBasedTopDocs(TRECQuery query, int kNN, float threshold, float lambda, int nCluster) throws Exception {
        
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        ScoreDoc[] hits = retrieveGeneral(query, 100);  // N = 100 i.e. #top docs
        //trecQueryparser.getAnalyzedQuery(query, 1);
        //ScoreDoc[] hits = getPOILevelContextualApproDocsGeneral(query, 100);
        //System.out.println("|||||||||||||||| INSIDE KNN hits len: " + hits.length);
        List<DocumentCluster> docCluster = new ArrayList<>();
        int id = 0;
        for (int i = 0; i < hits.length; ++i) {
            DocumentCluster temp = new DocumentCluster();
            temp.clusterId = id++;
            //System.out.println("getkNNDocs() for " + i + ": " + hits[i].doc);
            temp.docIds = getkNNDocs(query, hits[i].doc, hits, kNN, threshold);
            docCluster.add(temp);
        }

        int nDocCluster = docCluster.size();
        for (int i = 0; i < nDocCluster; ++i) {
            docCluster.get(i).clusterScore = getClusterScore(query, docCluster.get(i), lambda);
        }
        Collections.sort(docCluster, new cmpDocumentCluster());
        
//        for (int i = 0; i < nDocCluster; ++i) {
//            System.out.print(query.qid + ": " + docCluster.get(i).clusterId);
//            for (int j = 0; j < docCluster.get(i).docIds.length; ++j) {
//                System.out.print("\t" + docCluster.get(i).docIds[j]);
//            }
//            System.out.println("\t\tScore: " + docCluster.get(i).clusterScore);
//        }
//        System.exit(1);

        //System.out.println("||||||||| docCluster size: " + docCluster.size());
        int nClusterModified = Math.min(nCluster, hits.length);
        for (int i = 0; i < nClusterModified; ++i) {
        //for (int i = 0; i < nCluster; ++i) {
            for (int j = 0; j < docCluster.get(i).docIds.length; ++j) {
                //ScoreDoc sd = new ScoreDoc(getLuceneDocId(contextualQuery.get(contextualQueryIndex).posDocs[i]), score-=sub);
                ScoreDoc sd = new ScoreDoc(docCluster.get(i).docIds[j], 1.0f);
                if(docExist(hitsTemp, 0, sd) < 0)  // 2nd parameter '0' is garbage
                    hitsTemp.add(sd);
            }
        }
        
        if (hitsTemp.size() > 0) {
            hits = hitsTemp.toArray(new ScoreDoc[0]);
        }
        
        return hits;
    }
    
    // Gets 'topK' docs sorted (ascending / descending) based on their doc length
    public ScoreDoc[] getDocLenBasedTopDocs(TRECQuery query, int nHits, int topK) throws Exception {
        ScoreDoc[] hits = retrieveGeneral(query, nHits);
        ScoreDoc[] hitsUpdated = new ScoreDoc[hits.length];
        List<TermList> docs = new ArrayList<>();
        for (int i = 0; i < hits.length; ++i) {
            TermList temp = new TermList();
            temp.term = Integer.toString(hits[i].doc);
            temp.weight = (double) getDocLen(hits[i].doc);
            docs.add(temp);
        }
        
        Collections.sort(docs, new cmpTermListWeightAscending());
        //Collections.sort(docs, new cmpTermListWeight());
        
        for (int i = 0; i < hits.length; ++i) {
            ScoreDoc sd = new ScoreDoc(Integer.parseInt(docs.get(i).term), (float) docs.get(i).weight);
            hitsUpdated[i] = sd;
        }
        
        return hitsUpdated;
    }
    
    // Gets top 'n' docs that are contextually appropriate for the 'query' (~jointContext)
    public ScoreDoc[] getPOILevelContextualApproDocsGeneral(TRECQuery query, int nHits) throws Exception {
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        ScoreDoc[] hits = null;
        float fraction = (float) 1/nHits;
        float score = 1.0f, sub = 0.0f;
        
        String context = query.qid; // here qID ~ jointContext
        int contextualQueryIndex = getContextualQueryIndex(context);
        
        int n = Math.min(nHits, contextualQuery.get(contextualQueryIndex).posDocs.length);
        for (int i = 0; i < n; ++i) {
            //ScoreDoc sd = new ScoreDoc(getLuceneDocId(contextualQuery.get(contextualQueryIndex).posDocs[i]), 1.0f); // Adding top 'n' docs that are contextually appropriate, with equal scores 1.0
            ScoreDoc sd = new ScoreDoc(getLuceneDocId(contextualQuery.get(contextualQueryIndex).posDocs[i]), score-=sub);
            sub=fraction;
            hitsTemp.add(sd);
        }
        
        if (hitsTemp.size() > 0) {
            hits = hitsTemp.toArray(new ScoreDoc[0]);
        }
        
        return hits;
    }
    
    // Gets top 'n' docs that are contextually appropriate for the joint context available in 'query'
    public ScoreDoc[] getPOILevelContextualApproDocs(TRECQuery query, int nHits) throws Exception {
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        ScoreDoc[] hits = null;
        
        int treccsQueryJsonIndex = getTreccsQueryJsonIndex(query.qid);
        String context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-") + "-AND-City:-" + query.qcity;
        //String context = "Group-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).group.replace(" ", "-") + "-AND-Trip-type:-" + treccsQueryJson.get(treccsQueryJsonIndex).trip_type.replace(" ", "-") + "-AND-Trip-duration:-" + treccsQueryJson.get(treccsQueryJsonIndex).duration.replace(" ", "-");
        int contextualQueryIndex = getContextualQueryIndex(context);
        
        int n = Math.min(nHits, contextualQuery.get(contextualQueryIndex).posDocs.length);
        for (int i = 0; i < n; ++i) {
            ScoreDoc sd = new ScoreDoc(getLuceneDocId(contextualQuery.get(contextualQueryIndex).posDocs[i]), 1.0f); // Adding top 'n' docs that are contextually appropriate, with equal scores 1.0
            hitsTemp.add(sd);
        }
        
        if (hitsTemp.size() > 0) {
            hits = hitsTemp.toArray(new ScoreDoc[0]);
        }
        
        return hits;
    }

    // Get user preference docs from 'userPref'
    // modelChoice: 0 - docs with -ve rating (< +3)
    //              1 - docs with +ve rating (>= +3)
    // clusterId:   -1 - get the set of 'History' docs for the original query
    //              0, 1, ..., 10 - get the set of 'History' docs for 0th, 1st, ..., 10th sub-query

//    public WebDocSearcher_TRECCS_Novel() {
//    }
    public ScoreDoc[] getPrefDocs(String qID, int modelChoice, String clusterId) throws Exception {
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        ScoreDoc[] hits = null;
        int posNegThreshold = 3;
        
        int nUserPref = userPref.size();
        for (int i = 0; i < nUserPref; ++i) {
            //System.out.println(userPref.get(i).queryNo + " ||||||||||||||||||||||||||||||");
            if (userPref.get(i).queryNo == Integer.parseInt(qID)) {
                int nPref = userPref.get(i).nPreference;
                for (int j = 0; j < nPref; ++j) {
                    //System.out.print(userPref.get(i).docId[j] + " " + userPref.get(i).rating[j] + " ");
                    int luceneDocId = getLuceneDocId(userPref.get(i).docId[j]);
                    int rating = userPref.get(i).rating[j];
                    String prefCluster = userPref.get(i).clusterId[j];
                    if("-1".equals(clusterId)) {
                        if(modelChoice == 1) {
                            if (luceneDocId >= 0 && rating >= posNegThreshold) {
                                ScoreDoc sd = new ScoreDoc(getLuceneDocId(userPref.get(i).docId[j]), 1.0f); // Adding each doc from user pref. as scoredoc with equal score 1.0
                                hitsTemp.add(sd);
                            }
                        }
                        else {
                            if (luceneDocId >= 0 && rating < posNegThreshold) {
                                ScoreDoc sd = new ScoreDoc(getLuceneDocId(userPref.get(i).docId[j]), 1.0f); // Adding each doc from user pref. as scoredoc with equal score 1.0
                                hitsTemp.add(sd);
                            }
                        }                    
                    }
                    else {
                        if(modelChoice == 1) {
                            if (luceneDocId >= 0 && rating >= posNegThreshold && prefCluster.equals(clusterId)) {
                                ScoreDoc sd = new ScoreDoc(getLuceneDocId(userPref.get(i).docId[j]), 1.0f); // Adding each doc from user pref. as scoredoc with equal score 1.0
                                hitsTemp.add(sd);
                            }
                        }
                        else {
                            if (luceneDocId >= 0 && rating < posNegThreshold && prefCluster.equals(clusterId)) {
                                ScoreDoc sd = new ScoreDoc(getLuceneDocId(userPref.get(i).docId[j]), 1.0f); // Adding each doc from user pref. as scoredoc with equal score 1.0
                                hitsTemp.add(sd);
                            }
                        }                    

                    }
                }
                if (hitsTemp.size() > 0) {
                    //hits = retrieveCustomized(query, hitsTemp.size());
                    hits = hitsTemp.toArray(new ScoreDoc[0]);
                }
                break;
            }
        }
        return hits;
    }
    
    public ScoreDoc[] convert_finalListToHits(List<NewScore> finalList) throws Exception {
        List<ScoreDoc> hitsTemp = new ArrayList<>();
        ScoreDoc[] hits = null;
        
        int nfinalList = finalList.size();
        for (int i = 0; i < nfinalList; ++i) {
            ScoreDoc temp = new ScoreDoc(getLuceneDocId(finalList.get(i).docid), (float)finalList.get(i).score);
            //System.out.println(i + ": " + getLuceneDocId(finalList.get(i).docid) + " " + (float)finalList.get(i).score);
            hitsTemp.add(temp);
        }
        //System.exit(1);
        
        if (hitsTemp.size() > 0) {
            hits = hitsTemp.toArray(new ScoreDoc[0]);
        }
        return hits;
    }
    
    // Merges two HashMaps. Keep the max probability (p_w_given_R) for duplicate entries.
    public HashMap mergeHashMaps(HashMap<String, WordProbability> hashmap_1, HashMap<String, WordProbability> hashmap_2) throws Exception {
        HashMap<String, WordProbability> hashmap_merged = new LinkedHashMap<>();
        
        hashmap_merged.putAll(hashmap_1);
        
        for(Map.Entry<String, WordProbability> entry: hashmap_2.entrySet()) {
            //System.out.println(entry.getKey() + " : " + entry.getValue());
            if(hashmap_merged.containsKey(entry.getKey()))
                hashmap_merged.get(entry.getKey()).p_w_given_R = Math.max(hashmap_merged.get(entry.getKey()).p_w_given_R, entry.getValue().p_w_given_R);
            else
                hashmap_merged.put(entry.getKey(), new WordProbability(entry.getValue().w, entry.getValue().p_w_given_R));
        }

        return hashmap_merged;
    }
    
    // Merges two HashMaps. Sum up the probabilities (p_w_given_R) for duplicate entries.
    public HashMap mergeHashMapsSum(HashMap<String, WordProbability> hashmap_1, HashMap<String, WordProbability> hashmap_2) throws Exception {
        HashMap<String, WordProbability> hashmap_merged = new LinkedHashMap<>();
        
        hashmap_merged.putAll(hashmap_1);
        
        for(Map.Entry<String, WordProbability> entry: hashmap_2.entrySet()) {
            //System.out.println(entry.getKey() + " : " + entry.getValue());
            if(hashmap_merged.containsKey(entry.getKey()))
                hashmap_merged.get(entry.getKey()).p_w_given_R = hashmap_merged.get(entry.getKey()).p_w_given_R + entry.getValue().p_w_given_R;
            else
                hashmap_merged.put(entry.getKey(), new WordProbability(entry.getValue().w, entry.getValue().p_w_given_R));
        }

        return hashmap_merged;
    }
    
    // Returns updated (p_w_given_R /= (float) n) hashmap.
    public HashMap divideHashMap(HashMap<String, WordProbability> hashmap_1, int n) throws Exception {
        HashMap<String, WordProbability> hashmap_updated = new LinkedHashMap<>();
        hashmap_updated.putAll(hashmap_1);
        
        for(Map.Entry<String, WordProbability> entry: hashmap_updated.entrySet()) {
            entry.getValue().p_w_given_R /= (float) n;
        }

        return hashmap_updated;
    }
    
    // Returns hashmap of top 'n' terms
    public HashMap getTopTermsHashMap(HashMap<String, WordProbability> hashmap_1, int n) throws Exception {
        HashMap<String, WordProbability> hashmap_updated = new LinkedHashMap<>();
        List<TermList> terms = new ArrayList<>();
        
        for(Map.Entry<String, WordProbability> entry: hashmap_1.entrySet()) {
            TermList term = new TermList();
            term.term = entry.getValue().w;
            term.weight = entry.getValue().p_w_given_R;
            terms.add(term);
        }
        Collections.sort(terms, new cmpTermListWeight());
        
        int nTerms = Math.min(terms.size(), n);
        for (int i = 0; i < nTerms; ++i) {
            String term = terms.get(i).term;
            float weight = (float) terms.get(i).weight;
            hashmap_updated.put(term, new WordProbability(term, weight));
        }

        return hashmap_updated;
    }

    // Merge two distributions (HashMaps) in [α * PRM-H + (1-α) * PRM-R] fashion
    // PRM-H and PRM-R are term distributions generated from user history and top retrieved docs respectively
    public HashMap mergeDistributions(HashMap<String, WordProbability> hashmap_1, HashMap<String, WordProbability> hashmap_2, float alpha) throws Exception {
        HashMap<String, WordProbability> hashmap_merged = new LinkedHashMap<>();
        
        for(Map.Entry<String, WordProbability> entry: hashmap_1.entrySet()) {
            entry.getValue().p_w_given_R *= alpha;
        }
        for(Map.Entry<String, WordProbability> entry: hashmap_2.entrySet()) {
            entry.getValue().p_w_given_R *= (1.0f - alpha);
        }

        hashmap_merged.putAll(hashmap_1);
        
        for(Map.Entry<String, WordProbability> entry: hashmap_2.entrySet()) {
            //System.out.println(entry.getKey() + " : " + entry.getValue());
            if(hashmap_merged.containsKey(entry.getKey()))
                hashmap_merged.get(entry.getKey()).p_w_given_R = hashmap_merged.get(entry.getKey()).p_w_given_R + entry.getValue().p_w_given_R;
            else
                hashmap_merged.put(entry.getKey(), new WordProbability(entry.getValue().w, entry.getValue().p_w_given_R));
        }

        return hashmap_merged;
    }

    // Merge three distributions (HashMaps) in [α * PRM-H + β * PRM-R + (1-(α+β)) * PRM-R2] fashion
    public HashMap mergeDistributions3(HashMap<String, WordProbability> hashmap_1, HashMap<String, WordProbability> hashmap_2, HashMap<String, WordProbability> hashmap_3, float alpha, float beta) throws Exception {
        HashMap<String, WordProbability> hashmap_merged = new LinkedHashMap<>();
        
        for(Map.Entry<String, WordProbability> entry: hashmap_1.entrySet()) {
            entry.getValue().p_w_given_R *= alpha;
        }
        for(Map.Entry<String, WordProbability> entry: hashmap_2.entrySet()) {
            entry.getValue().p_w_given_R *= beta;
        }
        for(Map.Entry<String, WordProbability> entry: hashmap_3.entrySet()) {
            entry.getValue().p_w_given_R *= (1.0f - (alpha + beta));
        }

        hashmap_merged.putAll(hashmap_1);
        
        for(Map.Entry<String, WordProbability> entry: hashmap_2.entrySet()) {
            //System.out.println(entry.getKey() + " : " + entry.getValue());
            if(hashmap_merged.containsKey(entry.getKey()))
                hashmap_merged.get(entry.getKey()).p_w_given_R = hashmap_merged.get(entry.getKey()).p_w_given_R + entry.getValue().p_w_given_R;
            else
                hashmap_merged.put(entry.getKey(), new WordProbability(entry.getValue().w, entry.getValue().p_w_given_R));
        }
        for(Map.Entry<String, WordProbability> entry: hashmap_3.entrySet()) {
            //System.out.println(entry.getKey() + " : " + entry.getValue());
            if(hashmap_merged.containsKey(entry.getKey()))
                hashmap_merged.get(entry.getKey()).p_w_given_R = hashmap_merged.get(entry.getKey()).p_w_given_R + entry.getValue().p_w_given_R;
            else
                hashmap_merged.put(entry.getKey(), new WordProbability(entry.getValue().w, entry.getValue().p_w_given_R));
        }

        return hashmap_merged;
    }

    public TRECQuery updateQueryForFRLM(HashMap<String, WordProbability> hashmap_1, TRECQuery query) throws Exception {
        //System.out.println("\n||||||||||||||| Before ||||||||||||||||\n" + query.luceneQuery.toString(fieldToSearch) + "\n|||||||||||||||||||||||||||||||");
        TRECQuery qExp = new TRECQuery();
        qExp.qid = query.qid;
        qExp.qtitle = query.qtitle;
        qExp.luceneQuery = query.luceneQuery;
        qExp.fieldToSearch = query.fieldToSearch;
        qExp.qcity = query.qcity;
        qExp.qlat = query.qlat;
        qExp.qlng = query.qlng;
        qExp.qClass = query.qClass;
        qExp.qClassWeight = query.qClassWeight;
        
        qExp.qtitle = "";
        for(Map.Entry<String, WordProbability> entry: hashmap_1.entrySet()) {
            qExp.qtitle += entry.getValue().w + " ";
        }
        return qExp;
        //trecQueryparser.getAnalyzedQuery(query, 1);
        //System.out.println("\n||||||||||||||| After ||||||||||||||||\n" + query.luceneQuery.toString(fieldToSearch) + "\n|||||||||||||||||||||||||||||||");
        //System.exit(1);
    }

    // (hashmap_1 / hashmap_2)
    // i.e. Merges two HashMaps. Does P(t1) in hashmap_1 / P(t1) in hashmap_2 for duplicate terms.
    public HashMap divDistributions(HashMap<String, WordProbability> hashmap_1, HashMap<String, WordProbability> hashmap_2) throws Exception {
        HashMap<String, WordProbability> hashmap_merged = new LinkedHashMap<>();
        
        hashmap_merged.putAll(hashmap_1);
        
        for(Map.Entry<String, WordProbability> entry: hashmap_2.entrySet()) {
            //System.out.println(entry.getKey() + " : " + entry.getValue());
            if(hashmap_merged.containsKey(entry.getKey()))
                hashmap_merged.get(entry.getKey()).p_w_given_R = hashmap_merged.get(entry.getKey()).p_w_given_R / entry.getValue().p_w_given_R;
            else
                hashmap_merged.put(entry.getKey(), new WordProbability(entry.getValue().w, entry.getValue().p_w_given_R));
        }

        return hashmap_merged;
    }
    
    // Get the term (vector) from 'terms' that is in closest (cosine) proximity of 'vec'
    public String getClosestVec(List<TermList> terms, float[] vec, String qTerm1, String qTerm2) throws Exception {
        int n = terms.size();
        for (int i = 0; i < n; ++i) {
            int index = getW2VTermIndex(terms.get(i).term);
            if(index >= 0) {
                float sim = cosineSimilarity(vec, W2V.get(index).vector);
                terms.get(i).weight = sim;
            }
            else
                terms.get(i).weight = 0;
        }
        Collections.sort(terms, new cmpTermListWeight());
        for (int i = 0; i < n; ++i) {
            if(!terms.get(i).term.equals(qTerm1) && !terms.get(i).term.equals(qTerm2))
                return terms.get(i).term; // Closest term except query terms
        }
        //return terms.get(0).term;
        return "";
    }
    
    // Get additional query terms (composition Doi KDERLM CIKM '16) from topdocs (hits)
    public TRECQuery getComposition(TRECQuery query, ScoreDoc[] hits, int numFeedbackDocs) throws Exception {
                
        List<TermList> terms = new ArrayList<>();
        List<TermList> termsUniq = new ArrayList<>();
        int nDocs = Math.min(numFeedbackDocs, hits.length);
        for (int i = 0; i < nDocs; ++i) {
            String[] words = indexSearcher.doc(hits[i].doc).get(FIELD_FULL_BOW).split("\\s+");
            for (int j = 0; j < words.length; ++j) {
                TermList temp = new TermList();
                temp.term = words[j];
                terms.add(temp);
            }
        }
        termsUniq = getUniqTermList(terms);
        
        String composition = "";
        
        String[] qTerms = query.luceneQuery.toString(fieldToSearch).split("\\s+");
        for (int i = 0; i < qTerms.length-1; ++i) {
            int qTermIndex1 = getW2VTermIndex(qTerms[i]);
            if(qTermIndex1 >= 0) {
                int qTermIndex2 = getW2VTermIndex(qTerms[i+1]);
                if(qTermIndex2 >= 0) {
                    float[] qVec1 = W2V.get(qTermIndex1).vector;
                    float[] qVec2 = W2V.get(qTermIndex2).vector;
                    float[] qVecCompose = addVec(qVec1, qVec2);
                    composition += getClosestVec(termsUniq, qVecCompose, qTerms[i], qTerms[i+1]) + " "; // Additional terms
                }
                else
                    i++;
            }
        }        
//        System.out.println(query.luceneQuery.toString(fieldToSearch) + "\t" + composition);
//        System.exit(1);
        query.composition = composition;
        return query;
    }
    
    public int getQueryIndex(String qID) throws Exception {
        TRECQuery temp = new TRECQuery();
        temp.qid = qID;
        return Collections.binarySearch(queries, temp, new cmpTRECQuery());
    }
    
    public String[] getQueryTerms_1(String qID) throws Exception {
        int qIndex = getQueryIndex(qID);
        String[] qTerms = queries.get(qIndex).luceneQuery.toString(fieldToSearch).split("\\s+");
        return qTerms;
    }
    
    public String[] getQueryTerms(int qIndex) throws Exception {
        trecQueryparser.getAnalyzedQuery(queries.get(qIndex), 1);
        String[] qTerms = queries.get(qIndex).luceneQuery.toString(fieldToSearch).replace("(", "").replace(")", "").split("\\s+");
        return qTerms;
    }
    
    // Get DRMM histogram vector for the query-doc pair (qID, dpcID) considering there might be max 'qMax' #terms in query
    // i.e. we get qMax X 5 (#bins) dim vectors. Filled out with 0's where no. of qterms < 'qMax'
    public double[] getDRMMHistogram(int qIDIndex, int luceneDocId, int qMax, double exp) throws Exception {
        String[] qTerms = getQueryTerms(qIDIndex);
        List<TermList> docTerms = getTermsOnly(luceneDocId);
        int nDocTerms = docTerms.size();
        
        int dimMax = (qMax * 5) + 1; // Let max no. of query terms qMax=100, and for each queryterm-doc pair we get a histo of 5 bins. Concatenated to 100X5=500 dim vec. +1 i.e. last dim is for the geo part (e^(-x^2))
        double[] histoFinal = new double[dimMax];
        int k = 0;
        for (int i = 0; i < qTerms.length; ++i) {
            double IDF = getIdf(qTerms[i]);
            double[] histo = new double[5]; // DRMM histogram of 5 bins [(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0)]. Change this to [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0)], if cosineSim is in [0, 1].
            for (int j = 0; j < 5; ++j)
                histo[j] = 0; // initialize with 0. Possibly not needed in Java.
            for (int j = 0; j < nDocTerms; ++j) {
                float cosineSim = getCosineSimilarity(qTerms[i], docTerms.get(j).term);
                if(-1.0f <= cosineSim && cosineSim < -0.5f)
                    histo[0]++;
                else if(-0.5f <= cosineSim && cosineSim < 0.0f)
                    histo[1]++;
                else if(0.0f <= cosineSim && cosineSim < 0.5f)
                    histo[2]++;
                else if(0.5f <= cosineSim && cosineSim < 1.0f)
                    histo[3]++;
                else
                    histo[4]++;
            }
            for (int j = 0; j < 5; ++j) {
                //histoFinal[k++] = histo[j]; // CH: only count based
                if(histo[j] > 0)
                    //histoFinal[k++] = Math.log(histo[j]); // LCF: log(count)
                    histoFinal[k++] = Math.log(histo[j]) * IDF; // LCF * IDF
                else
                    histoFinal[k++] = histo[j];
            }
        }
        
        for (int i = k; i < dimMax - 1; ++i)
            histoFinal[i] = 0;
        
        // Adding dist(exp) part
        histoFinal[dimMax - 1] = exp;
        
        return histoFinal;
    }
    
    // getDRMMHistogram() in matrix format, concatinating 'exp' part (Geo dist) in each row
    // qMax = max #qterms, binSize = #bins
    public double[][] getDRMMHistogram_1(int qIDIndex, int luceneDocId, int qMax, int binSize, double exp) throws Exception {
        String[] qTerms = getQueryTerms(qIDIndex);
        List<TermList> docTerms = getTermsOnly(luceneDocId);
        int nDocTerms = docTerms.size();
        
        int row = qMax;
        int col = binSize + 1; // +1 for the geo part (e^(-x^2))
        double[][] histoFinal = new double[row][col];
        int k = 0;
        for (int i = 0; i < qTerms.length; ++i) {
            double IDF = getIdf(qTerms[i]);
            double[] histo = new double[binSize]; // DRMM histogram of 'binSize' bins
            for (int j = 0; j < binSize; ++j)
                histo[j] = 0; // initialize with 0. Possibly not needed in Java.
            for (int j = 0; j < nDocTerms; ++j) {
                float cosineSim = getCosineSimilarity(qTerms[i], docTerms.get(j).term);
                int binIndex = getHistogramBinIndex(-1.0, 1.0, binSize, cosineSim);
                histo[binIndex]++;                
            }
            for (int j = 0; j < binSize; ++j) {
                //histoFinal[k][j] = histo[j]; // CH: only count based
                if(histo[j] > 0)
                    //histoFinal[k][j] = Math.log(histo[j]); // LCF: log(count)
                    histoFinal[k][j] = Math.log(histo[j]) * IDF; // LCF * IDF
                else
                    histoFinal[k][j] = 0;
            }
            histoFinal[k][col-1] = exp;
            k++;
        }
        
        for (int i = k; i < row; ++i)
            for (int j = 0; j < col; ++j)
                histoFinal[i][j] = 0;

        return histoFinal;
    }
    
    public int getHistogramBinIndex(double min, double max, int nBin, double value) throws Exception {
        double binStep = (max - min) / nBin;
        double left = min;
        for (int i = 0; i < nBin; ++i) {
            if(left <= value && value < left + binStep)
                return i;
            else
                left += binStep;
        }
        return nBin - 1;
    }
    
    // Get DRMM histogram vectors (Geo POI, with e^(-x^2)) from qrels
    public void getGeoHistogram() throws Exception {
        String path1 = "/store/Data/TREC_CS/qrelsGeo_q_d_exp_rel.txt";
        String path2 = "/store/Data/TREC_CS/GeoDRMMData/";
        String path4 = "/store/Data/TREC_CS/TRECCS_MissingDdocs_inQrelsButNotinCorpus.txt";
        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        Collections.sort(queries, new cmpTRECQuery()); // Sorting queries by qIDs for binary searching
        String line = br.readLine();
        String[] content = line.split("\t");
        String qID = content[0];
        String docID = content[1];
        double exp = Double.parseDouble(content[2]);
        double relevance = Double.parseDouble(content[3]);
        String qIDcurrent = qID;
        int qIDIndex = getQueryIndex(qID);
        int luceneDocId = getLuceneDocId(docID);
        
        String path3;
        File file2;
        FileWriter writer;

        if(luceneDocId >= 0) {
            path3 = path2 + qID + "_" + docID + "_GeoDRMMHisto.txt";
            file2 = new File(path3);
            file2.createNewFile();
            writer = new FileWriter(file2, true);

            System.out.println("Writing file " + qID + "_" + docID + "_GeoDRMMHisto.txt");
            double[] histoGeo = getDRMMHistogram(qIDIndex, luceneDocId, 100, exp); // max no of qry terms = 100
            for (int i = 0; i < histoGeo.length; ++i) {
                writer.write(Double.toString(histoGeo[i]) + " ");
            }
            writer.close();
        }
        else {
            file2 = new File(path4);
            file2.createNewFile();
            writer = new FileWriter(file2, true);
            writer.write(docID + "\n");
            writer.close();
        }
        
        while ((line = br.readLine()) != null) {
            content = line.split("\t");
            qID = content[0];
            docID = content[1];
            exp = Double.parseDouble(content[2]);
            relevance = Double.parseDouble(content[3]);
            luceneDocId = getLuceneDocId(docID);
            
            if(luceneDocId >= 0) {
                path3 = path2 + qID + "_" + docID + "_GeoDRMMHisto.txt";
                file2 = new File(path3);
                file2.createNewFile();
                writer = new FileWriter(file2, true);

                if(!qID.equals(qIDcurrent)) {
                    qIDIndex = getQueryIndex(qID);
                    qIDcurrent = qID;
                }

                System.out.println("Writing file " + qID + "_" + docID + "_GeoDRMMHisto.txt");
                double[] histoGeo = getDRMMHistogram(qIDIndex, luceneDocId, 100, exp); // max no of qry terms = 100
                for (int i = 0; i < histoGeo.length; ++i) {
                    writer.write(Double.toString(histoGeo[i]) + " ");
                }
                writer.close();
            }
            else {
                file2 = new File(path4);
                file2.createNewFile();
                writer = new FileWriter(file2, true);
                writer.write(docID + "\n");
                writer.close();
            }
        }
        
        br.close(); fr.close();
    }
    
    // getGeoHistogram() for quey wise all docs from corpus
    public void getGeoHistogram_1() throws Exception {
        
        int binSize = 20; // #bins for Geo DRMM histogram
        
        String path1 = "/store/Data/TREC_CS/qID_lat_lng.txt";
        String path2 = "/store/Data/TREC_CS/TRECID_latLng.txt";
        String path3 = "/store/Data/TREC_CS/GeoDRMMData/";
        List<Geocode> users = new ArrayList<>();
        List<Geocode> POIs = new ArrayList<>();
        
        File file = new File(path1);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        
        while ((line = br.readLine()) != null) {
            String[] content = line.split(" ");
            String qID = content[0];
            float lat = Float.parseFloat(content[1]);
            float lng = Float.parseFloat(content[2]);
            Geocode temp = new Geocode();
            temp.ID = qID;
            temp.lat = lat;
            temp.lng = lng;
            users.add(temp);
        }
        Collections.sort(users, new cmpGeocode());
        br.close(); fr.close();
        
        file = new File(path2);
        fr= new FileReader(file);
        br = new BufferedReader(fr);
        
        while ((line = br.readLine()) != null) {
            String[] content = line.split(" ");
            String ID = content[0];
            float lat = Float.parseFloat(content[1]);
            float lng = Float.parseFloat(content[2]);
            Geocode temp = new Geocode();
            temp.ID = ID;
            temp.lat = lat;
            temp.lng = lng;
            POIs.add(temp);
        }
        Collections.sort(POIs, new cmpGeocode());
        br.close(); fr.close();
        
        Collections.sort(queries, new cmpTRECQuery()); // Sorting queries by qIDs for binary searching
        for (TRECQuery query : queries) {
            List<Geocode> docs = new ArrayList<>();
            System.out.println("nDocGlobal: " + nDocGlobal);
            for (int i = 0; i < nDocGlobal; ++i) {
                String city = indexSearcher.doc(i).get("cityId");
                if(city.equals(query.qcity)) {
                    int qIndex = getGeoIndex(query.qid, users);
                    int docIndex = getGeoIndex(indexSearcher.doc(i).get("docid"), POIs);
                    double dist;
                    if(docIndex >= 0)
                        dist = distance(users.get(qIndex).lat, POIs.get(docIndex).lat, users.get(qIndex).lng, POIs.get(docIndex).lng, 0.0, 0.0); // distance between user and POI
                    else
                        dist = -1;

                    Geocode temp = new Geocode();
                    temp.ID = indexSearcher.doc(i).get("docid");
                    temp.IDint = i;
                    temp.dist = dist;
                    docs.add(temp);
                }
            }
            int nDoc = docs.size();
            int qIDIndex = getQueryIndex(query.qid);
            double distMin = 9999, distMax = -9999;
            for (int i = 0; i < nDoc; ++i) { // Finding max dist, min dist
                if(docs.get(i).dist >= 0) {
                    distMin = docs.get(i).dist;
                    distMax = docs.get(i).dist;
                    for (int j = i+1; j < nDoc; ++j) {
                        if(docs.get(j).dist < distMin)
                            distMin = docs.get(j).dist;
                        if(docs.get(j).dist > distMax)
                            distMax = docs.get(j).dist;
                    }
                    break;
                }
            }

            for (int i = 0; i < nDoc; ++i) {
                double distNormalized;
                if(docs.get(i).dist >= 0)
                    distNormalized = (docs.get(i).dist - distMin) / (distMax - distMin);
                else
                    distNormalized = 0.5;
                double exp = Math.exp(-Math.pow(distNormalized, 2));
                double[][] histoGeo = getDRMMHistogram_1(qIDIndex, docs.get(i).IDint, 100, binSize, exp); // max no of qry terms = 100
                
                // Write the histogram for 'query'-'docs.get(i)' pair
                String path4 = path3 + query.qid + "_" + docs.get(i).ID + ".txt";
                file = new File(path4);
                file.createNewFile();
                FileWriter writer = new FileWriter(file, true);
                System.out.println("Writing file " + query.qid + "_" + docs.get(i).ID + ".txt\t" + query.qid + ":\tnDoc: " + nDoc + "\texp: " + exp);
                for (int j = 0; j < histoGeo.length; ++j) {
                    for (int k = 0; k < histoGeo[j].length; ++k)
                        writer.write(Double.toString(histoGeo[j][k]) + " ");
                    writer.write("\n");
                }
                writer.close();
            }
        }
    }
    
    public void splitTrainTest() throws Exception {
        int nAll = 610;
        int nTrain = 488; // 80% of 610
        
        String path1 = "/store/Data/TREC_CS/610_qIds";
        String path2 = "/store/Data/TREC_CS/train_qIds";
        String path3 = "/store/Data/TREC_CS/test_qIds";
        File file = new File(path1);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        
        List<String> qIds_all = new ArrayList<>();
        List<String> qIds_train = new ArrayList<>();
        List<String> qIds_test = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String[] content = line.split(" ");
            String qID = content[0];
            qIds_all.add(qID);
        }
        br.close(); fr.close();
        
        Random rand = new Random();
        int iTrain = 0;
        while(iTrain < nTrain) {
            int randIndex = rand.nextInt(nAll);
            if(idExist(qIds_train, qIds_all.get(randIndex)) < 0) {
                qIds_train.add(qIds_all.get(randIndex));
                iTrain++;
            }
        }
        Collections.sort(qIds_train);
        
        for (int i = 0; i < nAll; ++i) {
            if(Collections.binarySearch(qIds_train, qIds_all.get(i)) < 0)
                qIds_test.add(qIds_all.get(i));
        }
        Collections.sort(qIds_test);
        
        file = new File(path2);
        file.createNewFile();
        FileWriter writer = new FileWriter(file, true);
        int n = qIds_train.size();
        for (int i = 0; i < n; ++i)
            writer.write(qIds_train.get(i) + "\n");
        writer.close();
        
        file = new File(path3);
        file.createNewFile();
        writer = new FileWriter(file, true);
        n = qIds_test.size();
        for (int i = 0; i < n; ++i)
            writer.write(qIds_test.get(i) + "\n");
        writer.close();
    }
    
    // Get a list of (q_i_d_j) (q_i_d_k) 1/0 pairs for pairwise L2R (Geo DRMM Histo)
    public void getGeoHistogram_LTRPairs() throws Exception {
        String path1 = "/store/Data/TREC_CS/GeoDRMM_qID_pairs_rel.txt";
        String path2 = "/store/Data/TREC_CS/GeoDRMM_pairsForLTR.txt";
        
        String path3 = "/store/Data/TREC_CS/TRECCS_MissingDdocs_inQrelsButNotinCorpus_Uniq.txt";
        
        List<Qrel> q_d = new ArrayList<>();

        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        File file3 = new File(path3);
        FileReader fr3 = new FileReader(file3);
        BufferedReader br3 = new BufferedReader(fr3);
        
        String line;
        List<String> outCorpusDocs = new ArrayList<>();
        while ((line = br3.readLine()) != null) {
            outCorpusDocs.add(line);
        }
        Collections.sort(outCorpusDocs);
        
        line = br.readLine();
        String[] content = line.split(" ");
        String qID = content[0];
        String qID_docID = content[1];
        Double relevance = Double.parseDouble(content[2]);
        String qIDCurrent = qID;
        
        Qrel temp = new Qrel();
        temp.qID = qID;
        temp.docID = qID_docID;
        temp.relevance = relevance;
        q_d.add(temp);
        
        while ((line = br.readLine()) != null) {
            content = line.split(" ");
            qID = content[0];
            qID_docID = content[1];
            relevance = Double.parseDouble(content[2]);
            
            if(qID.equals(qIDCurrent)) {
                temp = new Qrel();
                temp.qID = qID;
                temp.docID = qID_docID;
                temp.relevance = relevance;
                q_d.add(temp);
            }
            else {
                int n = q_d.size();
                for (int i = 0; i < n; ++i) {
                    for (int j = i + 1; j < n; ++j) {
                        if(outCorpusDocExist(outCorpusDocs, q_d.get(i).docID, q_d.get(j).docID) == 0) {
                            System.out.println(q_d.get(i).qID + "\tPair: " + i + "\t" + j);
                            if(q_d.get(i).relevance > q_d.get(j).relevance)
                                writer.write(q_d.get(i).docID + "\t" + q_d.get(j).docID + "\t1\n");
                            else
                                writer.write(q_d.get(i).docID + "\t" + q_d.get(j).docID + "\t0\n");
                        }
                    }
                    
                }
                q_d = new ArrayList<>();
                temp = new Qrel();
                temp.qID = qID;
                temp.docID = qID_docID;
                temp.relevance = relevance;
                q_d.add(temp);
                qIDCurrent = qID;
            }
        }
        
        int n = q_d.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if(outCorpusDocExist(outCorpusDocs, q_d.get(i).docID, q_d.get(j).docID) == 0) {
                    System.out.println(q_d.get(i).qID + "\tPair: " + i + "\t" + j);
                    if (q_d.get(i).relevance > q_d.get(j).relevance)
                        writer.write(q_d.get(i).docID + "\t" + q_d.get(j).docID + "\t1\n");
                    else
                        writer.write(q_d.get(i).docID + "\t" + q_d.get(j).docID + "\t0\n");
                }
            }
        }
        writer.close();
        br.close(); fr.close();
        br3.close(); fr3.close();
    }
    
    public int outCorpusDocExist (List<String> outCorpusDocs, String q_d1, String q_d2) throws Exception {
        String[] content1 = q_d1.split("_");
        String[] content2 = q_d2.split("_");
        if(Collections.binarySearch(outCorpusDocs, content1[1]) >= 0)
            return 1;
        else if(Collections.binarySearch(outCorpusDocs, content2[1]) >= 0)
            return 1;
        else
            return 0;
    }
    
    // Get data in qrelsGeo_q_d_exp_rel.txt format.
    public void getData_for_GeoHistogram () throws Exception {
        String path1 = "/store/Data/TREC_CS/candidatePOIs";
        File file2 = new File(path1);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        for (int i = 0; i < nDocGlobal; ++i) {
            String city = indexSearcher.doc(i).get("cityId");
            if(city.equals("166") || city.equals("167") || city.equals("176") || city.equals("178") || city.equals("181") || city.equals("182") || city.equals("185") || city.equals("188") || city.equals("191") || city.equals("193") || city.equals("195") || city.equals("197") || city.equals("203") || city.equals("212") || city.equals("217") || city.equals("218") || city.equals("221") || city.equals("227") || city.equals("232") || city.equals("248") || city.equals("253") || city.equals("260") || city.equals("261") || city.equals("270") || city.equals("274") || city.equals("291") || city.equals("300") || city.equals("306") || city.equals("319") || city.equals("329") || city.equals("331") || city.equals("334") || city.equals("335") || city.equals("336") || city.equals("338") || city.equals("341") || city.equals("342") || city.equals("344") || city.equals("356") || city.equals("359") || city.equals("366") || city.equals("371") || city.equals("380") || city.equals("381") || city.equals("382") || city.equals("385") || city.equals("389") || city.equals("413"))
                writer.write(indexSearcher.doc(i).get(FIELD_ID) + "\n");
        }
        writer.close();
    }
    
    public int getGeoIndex (String ID, List<Geocode> IDs) throws Exception {
        Geocode temp = new Geocode();
        temp.ID = ID;
        return Collections.binarySearch(IDs, temp, new cmpGeocode());
    }

    // Initializing Relevance Model...
    public void initializeRLM(int numFeedbackDocs, int numFeedbackTerms, float QMIX) throws Exception {
        System.out.println("Initializing RLM... " + numFeedbackDocs + ", " + numFeedbackTerms + ", " + QMIX);
        numFeedbackDocsGlobal = numFeedbackDocs;
        numFeedbackTermsGlobal = numFeedbackTerms;
        QMIXGlobal = QMIX;
        String fieldForFeedback = fieldToSearch;        
        rlm = new RLM(indexReader, indexSearcher, analyzer, fieldForFeedback, numFeedbackDocs, numFeedbackTerms, QMIX, param1, W2Vmodel, W2V, bert, trecQueryparser, treccsQueryJson, contextualApproTerms);
    }
    
    public ScoreDoc[] RLMGeneralPP(TRECQuery query) throws Exception {
        ScoreDoc[] hits = null;
        TopDocs topDocs = null;
        HashMap<String, WordProbability> hashmap_PwGivenR;
        //hits = retrieveGeneral(query, numHits);
        trecQueryparser.getAnalyzedQuery(query, 1);
        hits = getPOILevelContextualApproDocsGeneral(query, numHits);
        
        // Re-ranking using KL-Div
        topDocs = new TopDocs(hits.length, hits, hits[0].score);
        rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
        hashmap_PwGivenR = rlm.RM3(query, topDocs);
        hits = retrieveGeneral(query, numHits);
        topDocs = new TopDocs(hits.length, hits, hits[0].score);
        hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

        
        topDocs = new TopDocs(hits.length, hits, hits[0].score);
        rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
        hashmap_PwGivenR = rlm.RM3(query, topDocs);
        //hashmap_PwGivenR = rlm.RM3_2General(query, topDocs);
        BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
        System.out.println("Re-retrieving with QE");
        System.out.println(booleanQuery.toString(fieldToSearch));
        hits = retrieveGeneralBooleanQuery(query, booleanQuery, numHits);

        // Re-ranking using KL-Div
//        topDocs = new TopDocs(hits.length, hits, hits[0].score);
//        hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

        return hits;
    }
    
    // Traditional RLM (RM3 version)
    public ScoreDoc[] RLMGeneral(TRECQuery query) throws Exception {
        ScoreDoc[] hits = null;
        TopDocs topDocs = null;
        HashMap<String, WordProbability> hashmap_PwGivenR;
        hits = retrieveGeneral(query, numHits);
//        trecQueryparser.getAnalyzedQuery(query, 1);
//        hits = getPOILevelContextualApproDocsGeneral(query, numHits);
        int kNN = 20;
        float threshold = 0.25f;
        float lambda = 0.4f;
        int nCluster = 2;
        //hits = getClusterBasedTopDocs(query, kNN, threshold, lambda, nCluster);
        
        //hits = getDocLenBasedTopDocs(query, numHits, 100);
        
//        System.out.println("#docs: " + hits.length);
//        for (int i = 0; i < hits.length; ++i) {
//            System.out.println(indexSearcher.doc(hits[i].doc).get(FIELD_ID));
//        }
//        System.exit(1);
        
        // Re-ranking using KL-Div
//        topDocs = new TopDocs(hits.length, hits, hits[0].score);
//        rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
//        hashmap_PwGivenR = rlm.RM3(query, topDocs);
//        hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

//        query = getComposition(query, hits, numFeedbackDocsGlobal);
//        //query.qtitle += " " + query.composition;
//        System.out.println("Composed Terms: " + query.composition);
        
        topDocs = new TopDocs(hits.length, hits, hits[0].score);
        rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split("\\s+"), 1);
        hashmap_PwGivenR = rlm.RM3(query, topDocs);
        //hashmap_PwGivenR = rlm.RM3_2General(query, topDocs);
        BooleanQuery booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
        System.out.println("Re-retrieving with QE");
        System.out.println(booleanQuery.toString(fieldToSearch));
        hits = retrieveGeneralBooleanQuery(query, booleanQuery, numHits);

        // Re-ranking using KL-Div
//        topDocs = new TopDocs(hits.length, hits, hits[0].score);
//        hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

        return hits;
    }
    
    // Exploring RM3...
    // modelChoice: 1 (PRM-H) Positive relevance model from user history
    // modelChoice: 2 (PRM-R) Positive relevance model from top retrieved documents
    // modelChoice: ...
    // alpha: To be used in the eqn. α * PRM-H + (1-α) * PRM-R
    public ScoreDoc[] RM3Explore(TRECQuery query, int modelChoice, float alpha, float beta) throws Exception {
        ScoreDoc[] hits = null;
        ScoreDoc[] hitsH = null;
        ScoreDoc[] hitsH_neg = null;
        ScoreDoc[] hitsR = null;
        
        TopDocs topDocs;
        TopDocs topDocsH;
        TopDocs topDocsH_neg;
        TopDocs topDocsR;
        
        List<TRECQuery> subQueries;
        int nSubQuery;
        
        HashMap<String, WordProbability> hashmap_PwGivenR;
        HashMap<String, WordProbability> hashmap_PwGivenR_2;    // 'hashmap_PwGivenR' for FRLM2
        HashMap<String, WordProbability> hashmap_PwGivenR_Merged;    // Merged 'hashmap_PwGivenR' and 'hashmap_PwGivenR_2' i.e. FRLM1 + FRLM2
        HashMap<String, WordProbability> hashmap_PwGivenR_H;
        HashMap<String, WordProbability> hashmap_PwGivenR_H_neg;
        HashMap<String, WordProbability> hashmap_PwGivenR_R;
        HashMap<String, WordProbability> hashmap_PwGivenR_R_2;
        
        HashMap<String, WordProbability> hashmap_PwGivenR_pos;
        HashMap<String, WordProbability> hashmap_PwGivenR_neg;
        
        TRECQuery queryExpanded, queryNegativeTags;
        BooleanQuery booleanQuery;
        
        UserPreference uPref = userPref.get(getUPrefIndex(userPref, Integer.parseInt(query.qid)));  // All user history docs for 'qID' query/user
        
        queryNegativeTags = getQueryNegativeTags(query); // Creating query 'queryNegativeTags' with -ve user preference tags
        trecQueryparser.getAnalyzedQuery(queryNegativeTags, 1);
        
        switch(modelChoice) {
            case 0: // Standard RM3
                    //hits = retrieveCustomized(query, numHits);
trecQueryparser.getAnalyzedQuery(query, 1);
hits = getPOILevelContextualApproDocsGeneral(query, numHits);
                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
                    //hashmap_PwGivenR = rlm.RM3(query, topDocs);
                    hashmap_PwGivenR = rlm.RM3_2(query, topDocs);
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
//                    // Re-ranking using KL-Div
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

////--------------------------------------------------
//                    // DG steps
//                    hits = retrieveCustomized(query, numHits);
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
//                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
//
//                    // Re-ranking using KL-Div
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
//                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
//
//                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
//                    System.out.println("Re-retrieving with QE");
//                    System.out.println(booleanQuery.toString(fieldToSearch));
//                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
////--------------------------------------------------

                    break;
                
            case 1: // PRM-H
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hits = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    if(hits == null || hits.length == 0)
                        //hits = retrieve(query);
                        hits = retrieveCustomized(query, numHits);
                    topDocs = new TopDocs(hits.length, hits, hits[0].score);

                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 0);   // using 'H'
                    //hashmap_PwGivenR = rlm.RM3(query, topDocs);
                    //hashmap_PwGivenR = rlm.RM3Customized(query, topDocs, uPref);
                    hashmap_PwGivenR = rlm.RM3Customized2(query, topDocs, uPref);
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
//                    // Re-ranking using KL-Div
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD                    

                
//--------------------------------------------------
//                    trecQueryparser.getAnalyzedQuery(query, 1);
//                    hits = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model
//                    if(hits == null || hits.length == 0)
//                        //hits = retrieve(query);
//                        hits = retrieveCustomized(query, numHits);
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//
//                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 0);   // using 'H'
//                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
//
//                    // Re-ranking using KL-Div
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//
//                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 0);   // using 'H'
//                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
//
////                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
////                    queryExpanded = query;
////                    queryExpanded.qtitle = booleanQuery.toString(fieldToSearch);
////                    queryExpanded.luceneQuery = booleanQuery;
//
//                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
//                    System.out.println("Re-retrieving with QE");
//                    System.out.println(booleanQuery.toString(fieldToSearch));
//                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
//--------------------------------------------------

                    break;

            case 2: // PRM-R
                    hits = retrieveCustomized(query, numHits);
                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
                    break;

            case 3: // NRM-H
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hits = getPrefDocs(query.qid, 0, "-1");   // 0 for -ve model. Gets -ve history.
                    if(hits == null || hits.length == 0)
                        //hits = retrieve(query);
                        hits = retrieveCustomized(query, numHits);
                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
                    break;

            case 5: // PRM-H (using multi-query)
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hits = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    if(hits == null || hits.length == 0)
                        //hits = retrieve(query);
                        hits = retrieveCustomized(query, numHits);
                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
                    
                    hashmap_PwGivenR = new LinkedHashMap<>();
                    subQueries = generateSubQueries(query);   // Generating multiple sub-queries
                    nSubQuery = subQueries.size();
                    for (int i = 0; i < nSubQuery; ++i) {
                        if(!"-1".equals(subQueries.get(i).qClass)) {
                            rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                            hashmap_PwGivenR = mergeHashMaps(hashmap_PwGivenR, rlm.RM3(query, topDocs));
                        }
                    }

                    
                    //rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "));
                    //hashmap_PwGivenR = rlm.RM3(query, topDocs);
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
                    break;

            case 6: // α * PRM-H + (1-α) * PRM-R [using multi-query] Ani...
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    hitsR = retrieveCustomized(query, numHits);
                    
                    if(hitsH == null || hitsH.length == 0)
                        //hits = retrieve(query);
                        //hitsH = retrieveCustomized(query, numHits);
                        hitsH = hitsR;
                    topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    
                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    subQueries = generateSubQueries(query);   // Generating multiple sub-queries
                    nSubQuery = subQueries.size();
                    for (int i = 0; i < nSubQuery; ++i) {
                        if(!"-1".equals(subQueries.get(i).qClass)) {
                            rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                            hashmap_PwGivenR_H = mergeHashMaps(hashmap_PwGivenR_H, rlm.RM3(query, topDocsH));
                            rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                            hashmap_PwGivenR_R = mergeHashMaps(hashmap_PwGivenR_R, rlm.RM3(query, topDocsR));
                        }
                    }

                    hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                    
                    query.qtitle = "";
                    for(Map.Entry<String, WordProbability> entry: hashmap_PwGivenR.entrySet()) {
                        query.qtitle += entry.getValue().w + " ";
                    }
                    //System.out.println("HashMap Len: " + hashmap_PwGivenR.size());
                    
                    hits = retrieveCustomized(query, numHits);
                    topDocs = new TopDocs(hits.length, hits, hits[0].score);

                    
                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_H, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
                    break;

            case 7: // α * PRM-H + (1-α) * PRM-R [using multi-query] Debu...
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    //hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model
                    hashmap_PwGivenR = new LinkedHashMap<>();
                    
                    subQueries = generateSubQueries(query);   // Generating multiple sub-queries
                    nSubQuery = subQueries.size();
                    for (int i = 0; i < nSubQuery; ++i) {
                        if(!"-1".equals(subQueries.get(i).qClass)) {
                            //hitsH = getPrefDocs(query.qid, 1);   // 1 for +ve model
                            hitsR = retrieveCustomized(subQueries.get(i), numHits);
                            hitsH = getPrefDocs(query.qid, 1, subQueries.get(i).qClass);   // 1 for +ve model. Gets +ve history.
                            
                            if(hitsR != null && hitsH != null && hitsR.length > 0 && hitsH.length > 0) {
                                    topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);

                                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                                    rlm.setFeedbackStatsDirect(topDocsH, subQueries.get(i).luceneQuery.toString(fieldToSearch).split(" "), 1);
                                    //hashmap_PwGivenR_H = mergeHashMaps(hashmap_PwGivenR_H, rlm.RM3(subQueries.get(i), topDocsH));
                                    hashmap_PwGivenR_H = rlm.RM3(subQueries.get(i), topDocsH);
                                    rlm.setFeedbackStatsDirect(topDocsR, subQueries.get(i).luceneQuery.toString(fieldToSearch).split(" "), 1);
                                    //hashmap_PwGivenR_R = mergeHashMaps(hashmap_PwGivenR_R, rlm.RM3(subQueries.get(i), topDocsR));
                                    hashmap_PwGivenR_R = rlm.RM3(subQueries.get(i), topDocsR);

                                    hashmap_PwGivenR = mergeHashMaps(hashmap_PwGivenR, mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha));    
                            }
                            
                        }
                    }
                    
                    if(hashmap_PwGivenR.size() > 0) {
                        query.qtitle = "";
                        for (Map.Entry<String, WordProbability> entry : hashmap_PwGivenR.entrySet()) {
                            query.qtitle += entry.getValue().w + " ";
                        }                        
                    }
                    //System.out.println("HashMap Len: " + hashmap_PwGivenR.size());
                    System.out.println("||||||||||||||| ANI Query: " + query.qtitle + "\nEnd of ANI\n");
//                    for (Map.Entry<String, WordProbability> entry : hashmap_PwGivenR.entrySet()) {
//                        System.out.println(entry.getValue().w + " " + entry.getValue().p_w_given_R);
//                    }

                    hits = retrieveCustomized(query, numHits);
                    //hits = retrieveCustomized(getExpandedQueryW2VKDE(query, 5, 100), numHits);
                    
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    rlm.setFeedbackStatsDirect(topDocs, query.luceneQuery.toString(fieldToSearch).split(" "));
//                    hashmap_PwGivenR = rlm.RM3(query, topDocs);
//                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
//                    System.out.println("Re-retrieving with QE");
//                    System.out.println(booleanQuery.toString(fieldToSearch));
//                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);

                    
                    break;
                
            case 8: // H_Mix * PRM-H + (1 - H_Mix) * PRM-R      Single query with +ve rated (+3, +4 only) docs in H set
                    // Only +ve model i.e. taking the docs with rating +3, +4 only, into consideration
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    hitsR = retrieveCustomized(query, numHits);
//                    TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
//                    hitsR = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);

                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    //System.exit(1);
                    
                    // Re-ranking using KL-Div
//                    hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_R, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_R, hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
//                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
//                    hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);

                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
                        hashmap_PwGivenR_H = rlm.RM3Customized(query, topDocsH, uPref);
                        
                        hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                    }
                    else
                        hashmap_PwGivenR = hashmap_PwGivenR_R;
                    
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
//                    // Re-ranking using KL-Div
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

                    break;

            case 9: // H_Mix * PRM-H + (1 - H_Mix) * PRM-R      Single query with +ve rated (+3, +4 only) docs in H set
                    // Mixing +ve model and -ve model i.e. considering docs with rating +3, +4 as +ve and docs with rating < +3 as -ve.
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    hitsH_neg = getPrefDocs(query.qid, 0, "-1");   // 0 for -ve model. Gets -ve history.
                    hitsR = retrieveCustomized(query, numHits); // Initial retrieval

                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_H_neg = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();

                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);    // Initial TopDocs 'topDocsR' based on initial top retrieved docs 'hitsR'
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);  // Initial hashmap based on 'topDocsR'

                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);    // Initial TopDocs 'topDocsH' based on user's +ve history 'hitsH'
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);  // Initial hashmap based on 'topDocsH'
                        
                        // Merged two hashmaps 'hashmap_PwGivenR_H' and 'hashmap_PwGivenR_R' using 'alpha' which is H_Mix1 we say. Merged hashmap is a +ve hashmap 'hashmap_PwGivenR_pos' which is basically H_Mix1 * PRM-H + (1 - H_Mix1) * PRM-R
                        hashmap_PwGivenR_pos = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                    }
                    else
                        hashmap_PwGivenR_pos = hashmap_PwGivenR_R;
                    
                    // Re-retrieving with the +ve hashmap 'hashmap_PwGivenR_pos'
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_pos, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    topDocs = new TopDocs(hits.length, hits, hits[0].score);    // New TopDocs after Re-retrieving with Query Expansion
                    
                    // Taking the expanded query into consideration
                    queryExpanded = query;
                    queryExpanded.qtitle = booleanQuery.toString(fieldToSearch);
                    queryExpanded.luceneQuery = booleanQuery;
                    
                    // Again RM3 to update the hashmap 'hashmap_PwGivenR_pos'
                    rlm.setFeedbackStatsDirect(topDocs, queryExpanded.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    hashmap_PwGivenR_pos = rlm.RM3(queryExpanded, topDocs);
                    
                    // Merged 'hashmap_PwGivenR_pos' and 'hashmap_PwGivenR_H' using 'beta' which is H_Mix2 we say, the same way as before.
                    if(hitsH != null && hitsH.length > 0) {
                        hashmap_PwGivenR_pos = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_pos, beta);
                    }

                    // Again RM3 to generate the -ve hashmap 'hashmap_PwGivenR_neg' based on 'topDocsH_neg' which is basically user's -ve history 'hitsH_neg'
                    if(hitsH_neg != null && hitsH_neg.length > 0) {
                        topDocsH_neg = new TopDocs(hitsH_neg.length, hitsH_neg, hitsH_neg[0].score);    // TopDocs 'topDocsH_neg' based on user's -ve history 'hitsH_neg'
                        rlm.setFeedbackStatsDirect(topDocsH_neg, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        hashmap_PwGivenR_H_neg = rlm.RM3(query, topDocsH_neg);  // hashmap based on 'topDocsH_neg'
                    }

                    hits = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_pos, hashmap_PwGivenR_H_neg, query, topDocs));  // Re-ranking using KLD

                    break;

            case 10: // FRLM as in SIGIR 2019
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    //hitsR = retrieveCustomized(query, numHits);
                    
                    //topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    

                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    
                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
                        hashmap_PwGivenR_H = rlm.RM3Customized(query, topDocsH, uPref);
                        
                        System.out.println("\n||||||||||||||| Before ||||||||||||||||\n" + query.luceneQuery.toString(fieldToSearch) + "\n|||||||||||||||||||||||||||||||");
                        queryExpanded = updateQueryForFRLM(hashmap_PwGivenR_H, query);
                        //trecQueryparser.getAnalyzedQuery(queryExpanded, 1);
                        hitsR = retrieveCustomized(queryExpanded, numHits);
                        topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                        
                        System.out.println("\n||||||||||||||| After ||||||||||||||||\n" + queryExpanded.luceneQuery.toString(fieldToSearch) + "\n|||||||||||||||||||||||||||||||");
                        System.out.println("\nhitsR len: " + hitsR.length);
//                        for (int i = 0; i < hitsR.length; ++i) {
//                            System.out.println("Doc " + i + ": " + hitsR[i].doc);
//                        }
                        
                        rlm.setFeedbackStatsDirect(topDocsR, queryExpanded.luceneQuery.toString(fieldToSearch).split(" "), 1);
                        hashmap_PwGivenR_R = rlm.RM3(queryExpanded, topDocsR);
                        
                        hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                    }
                    else {
                        hitsR = retrieveCustomized(query, numHits);
                        topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                        rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                        hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                        hashmap_PwGivenR = hashmap_PwGivenR_R;
                    }
                    
                    
                    System.out.println("\n--------------------- FINAL hashmap_PwGivenR ---------------------");
                    for(Map.Entry<String, WordProbability> entry: hashmap_PwGivenR.entrySet()) {
                        query.qtitle += entry.getValue().w + " ";
                        System.out.print(entry.getValue().w + "(" + entry.getValue().p_w_given_R + ")" + " ");
                    }
                    System.out.println("\n--------------------- END of hashmap_PwGivenR ---------------------\n");
 
                    
//                    // Re-ranking using KL-Div
//                    hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_R, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_R, hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
//                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
//                    hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);

//                    if(hitsH != null && hitsH.length > 0) {
//                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
//                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
//                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
//                        hashmap_PwGivenR_H = rlm.RM3Customized(query, topDocsH, uPref);
//                        
//                        hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
//                    }
//                    else
//                        hashmap_PwGivenR = hashmap_PwGivenR_R;
                    
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
//                    // Re-ranking using KL-Div
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

                    break;
                
            case 11: // FRLM2
                    // Only +ve model i.e. taking the docs with rating +3, +4 only, into consideration
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    hitsR = retrieveCustomized(query, numHits);
                    //hitsR = getPOILevelContextualApproDocsGeneral(query, numHits);
                    //hitsR = getPOILevelContextualApproDocs(query, numHits);
//                    TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
//                    hitsR = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);

                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    
                    //initializeRLM(7, 20, 0.4f);
                    
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    hashmap_PwGivenR_R = rlm.RM3_2(query, topDocsR);
                    //System.exit(1);
                    
                    // Re-ranking using KL-Div
//                    hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_R, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_R, hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
//                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
//                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
//                    hashmap_PwGivenR_R = rlm.RM3_2(query, topDocsR);
                    
                    //initializeRLM(7, 30, 0.4f);

                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
                        //hashmap_PwGivenR_H = rlm.RM3Customized(query, topDocsH, uPref);
                        hashmap_PwGivenR_H = rlm.RM3Customized2(query, topDocsH, uPref);
                        
                        hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                    }
                    else
                        hashmap_PwGivenR = hashmap_PwGivenR_R;
                    
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
//                    // Re-ranking using KL-Div
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

                    break;

            case 12: // FRLM1 + FRLM2
                    // FRLM1 part
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    hitsR = retrieveCustomized(query, numHits);
                    //hitsR = getPOILevelContextualApproDocs(query, numHits);
//                    TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
//                    hitsR = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);

                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    //System.exit(1);
                    
//                    // Re-ranking using KL-Div
//                    hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_R, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_R, hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
//                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
//                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
//                    hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);

                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
                        hashmap_PwGivenR_H = rlm.RM3Customized(query, topDocsH, uPref);
                        
                        hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                    }
                    else
                        hashmap_PwGivenR = hashmap_PwGivenR_R;
                    // End of FRLM1 part
                    
                    
                    // FRLM2 part
                    initializeRLM(7, 45, 0.2f);
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    //hitsR = retrieveCustomized(query, numHits);
                    hitsR = getPOILevelContextualApproDocs(query, numHits);
//                    TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
//                    hitsR = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);

                    hashmap_PwGivenR_H = new LinkedHashMap<>();
                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    hashmap_PwGivenR_R = rlm.RM3_2(query, topDocsR);
                    //System.exit(1);
                    
                    // Re-ranking using KL-Div
                    hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_R, query, topDocsR));  // Re-ranking using KLD
                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_R, hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    hashmap_PwGivenR_R = rlm.RM3_2(query, topDocsR);

                    
                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
                        hashmap_PwGivenR_H = rlm.RM3Customized2(query, topDocsH, uPref);
                        
                        hashmap_PwGivenR_2 = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                    }
                    else
                        hashmap_PwGivenR_2 = hashmap_PwGivenR_R;
                    // End of FRLM2 part

                    // FRLM1 + FRLM2
                    hashmap_PwGivenR_Merged = mergeDistributions(hashmap_PwGivenR, hashmap_PwGivenR_2, beta);
                    
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR_Merged, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
//                    // Re-ranking using KL-Div
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

                    break;

            case 13: // H+ + TopDocs + contextualApproDocs
                    // Only +ve model i.e. taking the docs with rating +3, +4 only, into consideration
                    trecQueryparser.getAnalyzedQuery(query, 1);
                    hitsH = getPrefDocs(query.qid, 1, "-1");   // 1 for +ve model. Gets +ve history.
                    //hitsR = retrieveCustomized(query, numHits);
                    hitsR = retrieveCustomized(query, numHits);
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    hashmap_PwGivenR_R = rlm.RM3_2(query, topDocsR);


                    hitsR = getPOILevelContextualApproDocs(query, numHits);
//                    TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f);
//                    hitsR = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);

//                    hashmap_PwGivenR_H = new LinkedHashMap<>();
//                    hashmap_PwGivenR_R = new LinkedHashMap<>();
                    
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);
                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    hashmap_PwGivenR_R_2 = rlm.RM3_2(query, topDocsR);
                    //System.exit(1);
                    
                    // Re-ranking using KL-Div
                    hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_R_2, query, topDocsR));  // Re-ranking using KLD
                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
                    //hitsR = convert_finalListToHits(rlm.rerankUsingRBLM_PosNeg(hashmap_PwGivenR_R, hashmap_PwGivenR_H_neg, query, topDocsR));  // Re-ranking using KLD
                    topDocsR = new TopDocs(hitsR.length, hitsR, hitsR[0].score);
                    rlm.setFeedbackStatsDirect(topDocsR, query.luceneQuery.toString(fieldToSearch).split(" "), 1);   // using 'R'
                    //hashmap_PwGivenR_R = rlm.RM3(query, topDocsR);
                    hashmap_PwGivenR_R_2 = rlm.RM3_2(query, topDocsR);
                    


                    if(hitsH != null && hitsH.length > 0) {
                        topDocsH = new TopDocs(hitsH.length, hitsH, hitsH[0].score);
                        rlm.setFeedbackStatsDirect(topDocsH, query.luceneQuery.toString(fieldToSearch).split(" "), 0);
                        //hashmap_PwGivenR_H = rlm.RM3(query, topDocsH);
                        //hashmap_PwGivenR_H = rlm.RM3Customized(query, topDocsH, uPref);
                        hashmap_PwGivenR_H = rlm.RM3Customized2(query, topDocsH, uPref);
                        
                        //hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_H, hashmap_PwGivenR_R, alpha);
                        hashmap_PwGivenR = mergeDistributions3(hashmap_PwGivenR_H, hashmap_PwGivenR_R, hashmap_PwGivenR_R_2, alpha, beta);
                    }
                    else
                        hashmap_PwGivenR = hashmap_PwGivenR_R;
                        //hashmap_PwGivenR = mergeDistributions(hashmap_PwGivenR_R, hashmap_PwGivenR_R_2, alpha);
                    
                    booleanQuery = rlm.getExpandedQueryDirect(hashmap_PwGivenR, query);
                    System.out.println("Re-retrieving with QE");
                    System.out.println(booleanQuery.toString(fieldToSearch));
                    hits = retrieveCustomizedBooleanQuery(query, booleanQuery, numHits);
                    
//                    // Re-ranking using KL-Div
//                    topDocs = new TopDocs(hits.length, hits, hits[0].score);
//                    hits = convert_finalListToHits(rlm.rerankUsingRBLM(hashmap_PwGivenR, query, topDocs));  // Re-ranking using KLD

                    break;
                
        }
        return hits;
    }
    
    public void getParsedQuery () throws Exception {
        String path = "/store/Data/TREC_CS/tagPartEdited";
        File file = new File(path);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        int i = 0;
        while ((line = br.readLine()) != null) {
            TRECQuery q = new TRECQuery();
            q.qtitle = line;
            trecQueryparser.getAnalyzedQuery(q, 1);
            System.out.println(q.luceneQuery.toString(fieldToSearch));
        }
        System.exit(1);
        br.close(); fr.close();


    }
    
    public int idExist (List<String> list, String ID) throws Exception {
        int n = list.size();
        for (int i = 0; i < n; ++i) {
            if(ID.equals(list.get(i)))
                return i;
        }
        return -1;
    }
    
    public int idExist (String[] a, String ID) throws Exception {
        for (int i = 0; i < a.length; ++i) {
            if(ID.equals(a[i]))
                return i;
        }
        return -1;
    }
    
    public int getUnion (String[] a, String[] b) throws Exception {
        
        int counter = a.length;
        for (int i = 0; i < b.length; ++i) {
            if(idExist(a, b[i]) < 0)
                counter++;
        }
        return counter;
    }
    
    public int getIntersection (String[] a, String[] b) throws Exception {
        
        int counter = 0;
        for (int i = 0; i < b.length; ++i) {
            if(idExist(a, b[i]) >= 0)
                counter++;
        }
        return counter;
    }
    
    // Exists in 'b', but not in 'a'
    public int getInvIntersection (String[] a, String[] b) throws Exception {
        
        int counter = 0;
        for (int i = 0; i < b.length; ++i) {
            if(idExist(a, b[i]) == -1)
                counter++;
        }
        return counter;
    }
    
    public int getIndex (int[] a, int num) throws Exception {
        for (int i = 0; i < a.length; i++) {
            if(a[i] == num)
                return i;
        }
        return -1;
    }
    
    public int[] updateRank (int[] a) throws Exception {
        int[] sorted = a.clone();
        Arrays.sort(sorted);
        Rank temp = new Rank();
        temp.original = a;
        temp.updated = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            int index = getIndex(a, sorted[i]);
            temp.updated[index] = i;
        }
        return temp.updated;
    }
    
//    // return Kendall tau distance between two permutations
//    public static long distance(int[] a, int[] b) {
//        if (a.length != b.length) {
//            throw new IllegalArgumentException("Array dimensions disagree");
//        }
//        int n = a.length;
//
//        int[] ainv = new int[n];
//        for (int i = 0; i < n; i++)
//            ainv[a[i]] = i;
//
//        Integer[] bnew = new Integer[n];
//        for (int i = 0; i < n; i++)
//            bnew[i] = ainv[b[i]];
//
//        return Inversions.count(bnew);
//    }
    
    public float getKendalsTau (String[] a, String[] b) throws Exception {
        
        int counter = 0;
        for (int i = 0; i < b.length; ++i) {
            if(idExist(a, b[i]) >= 0)
                counter++;
        }
        
        String[] intersection = new String[counter];
        int j = 0;
        for (int i = 0; i < b.length; ++i) {
            if(idExist(a, b[i]) >= 0)
                intersection[j++] = b[i];
        }
        int aRank[] = new int[counter];
        int bRank[] = new int[counter];
        for (int i = 0; i < counter; ++i) {
            aRank[i] = idExist(a, intersection[i]);
            bRank[i] = idExist(b, intersection[i]);
        }
        
//        aRank[0] = 5;
//        aRank[1] = 3;
//        aRank[2] = 2;
        
        int aRankUpdated[] = updateRank(aRank);
        int bRankUpdated[] = updateRank(bRank);
        
//        System.out.println("counter: " + counter);
//        for (int i = 0; i < counter; ++i) {
//            System.out.println(intersection[i] + "\t" + aRank[i] + "\t" + bRank[i]);
//        }
//        System.out.println("-----------------------------------------");
//        for (int i = 0; i < counter; ++i) {
//            System.out.println(intersection[i] + "\t" + aRankUpdated[i] + "\t" + bRankUpdated[i]);
//        }
        
        
        
        // inverse of bRank
        int[] inv = new int[counter];
        for (int i = 0; i < counter; i++)
            inv[bRankUpdated[i]] = i;
        // calculate Kendall tau distance
        int tau = 0;
        for (int i = 0; i < counter; i++) {
            for (j = i+1; j < counter; j++) {
                // check if p[i] and p[j] are inverted
                if (inv[aRankUpdated[i]] > inv[aRankUpdated[j]])
                    tau++;
            }
        }
        
        float normalizedTau = (float) tau / (counter * (counter-1) / 2);
        
        //System.out.println("tau: " + tau + "\t" + normalizedTau);
        return normalizedTau;
    }
    
    public void analyseResults () throws Exception {
        
        int nTop = 10;
        
        String bm25HitsFilrPath = "/store/Data/TRECAdhoc/querywise_topDocsIDs_BM25";
        //String bm25HitsFilrPath = "/store/Data/TRECAdhoc/querywise_topDocsIDs_BM25_TRECRb";
        String kNNDocsPath = "/store/Data/TRECAdhoc/kNNDocsTRECRb.txt";
        
        List<ContextualQuery> contextualQueryBM25 = new ArrayList<>();

        File file = new File(bm25HitsFilrPath);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
                        
        while ((line = br.readLine()) != null) {
            String[] splited = line.split(" ");
            ContextualQuery temp = new ContextualQuery();
            temp.context = splited[0];
            temp.posDocs = new String[1000];
            int j = 0;
            for (int i = 1; i < splited.length; ++i) {
                temp.posDocs[j++] = splited[i];
            }
            contextualQueryBM25.add(temp);
        }
        br.close(); fr.close();

        Collections.sort(contextualQueryBM25, new cmpContextQuery());
        
        int qIndex = 0;
        int nContextualQuery = contextualQuery.size();
        //for (int i = 0; i < 99; ++i) {
        for (int i = 0; i < nContextualQuery; ++i) {
            
            String[] bm25Hits = new String[nTop];
            String[] rlmHits = new String[nTop];
            String[] kNNHits = new String[nTop];
            
            //System.out.print("BM25: " + contextualQueryBM25.get(i).context);
            int nDocs = nTop;//contextualQueryBM25.get(i).posDocs.length;
            for (int j = 0; j < nDocs; ++j) {
                //System.out.print("\t" + contextualQueryBM25.get(i).posDocs[j]);
                bm25Hits[j] = contextualQueryBM25.get(i).posDocs[j];
            }
            //System.out.println();
            
            //System.out.print("RLM: " + contextualQuery.get(i).context);
            nDocs = nTop;//contextualQuery.get(i).posDocs.length;
            for (int j = 0; j < nDocs; ++j) {
                //System.out.print("\t" + contextualQuery.get(i).posDocs[j]);
                rlmHits[j] = contextualQuery.get(i).posDocs[j];
            }
            
//        int kNN = 20;
//        float threshold = 0.25f;
//        float lambda = 0.4f;
//        int nCluster = 100;
//        ScoreDoc[] hits = getClusterBasedTopDocs(queries.get(qIndex++), kNN, threshold, lambda, nCluster);
//            for (int j = 0; j < nDocs; ++j) {
//                kNNHits[j] = indexSearcher.doc(hits[j].doc).get(FIELD_ID);
//            }
            
            List<ContextualQuery> contextualQuerykNN = getkNNDocs(kNNDocsPath);
            for (int j = 0; j < nDocs; ++j) {
                //System.out.print("\t" + contextualQuery.get(i).posDocs[j]);
                kNNHits[j] = contextualQuerykNN.get(i).posDocs[j];
            }
            //System.out.println();
            
            //int interSection = getIntersection(bm25Hits, rlmHits);
            int interSection = getInvIntersection(bm25Hits, rlmHits);
            //int union = getUnion(bm25Hits, rlmHits);
            //int interSection = getInvIntersection(bm25Hits, kNNHits);
            //System.out.println(contextualQueryBM25.get(i).context + ": Intersection: " + getIntersection(bm25Hits, rlmHits) + "\tUnion: " + getUnion(bm25Hits, rlmHits));
            //System.out.println(contextualQueryBM25.get(i).context + "\t" + interSection + "\t" + union + "\t" + (float) interSection/union);
            //System.out.println(contextualQueryBM25.get(i).context + "\t" + (float) interSection/union);
            //System.out.println((float) interSection/union);
            System.out.println((float) interSection/nTop);
            
            //System.out.println(contextualQueryBM25.get(i).context + "\t" + getKendalsTau(bm25Hits, rlmHits));
            //System.out.println(getKendalsTau(bm25Hits, rlmHits));

            //System.exit(1);
            
        }
        
        
        
        
    }
    
    public void removeMultitermVecs() throws Exception {
        String path1 = "/store/Data/TREC_CS/transformers/BERT_vectors.txt";
        //String path1 = "/store/Data/TREC_CS/transformers/p";
        String path2 = "/store/Data/TREC_CS/transformers/BERT_vectors1.txt";
        File file1 = new File(path1);
        FileReader fr = new FileReader(file1);
        BufferedReader br = new BufferedReader(fr);
        
        File file2 = new File(path2);
        file2.createNewFile();
        FileWriter writer = new FileWriter(file2, true);
        
        String line;
        
//        line = br.readLine();
//        String[] content = line.split("\t");
//        String qID = content[0];
//        int rel = Integer.parseInt(content[3]);
//        
//        String qID1;
//        int relSum = 0;
//        qID1 = qID;
//        
//        if(rel >= 1.0f)
//            relSum ++;
//        
        while ((line = br.readLine()) != null) {
            String[] content = line.split(" ");
            String term = content[1];
            boolean numeric = true;

            try {
                Double num = Double.parseDouble(term);
            } catch (NumberFormatException e) {
                numeric = false;
            }
            if(numeric)
                writer.write(line + "\n");
        }
        br.close(); fr.close();
        
        writer.close();
        
    }

    public void retrieveAll() throws Exception {
        
        collectionSizeGlobal = getCollectionSize();
        nDocGlobal = indexReader.maxDoc();
        
        //PP();
        //removeMultitermVecs();
        //parseGloveTerms();
        //mergeGloveTerms();
        
//        int docId = 2222;
//        List<String> para = getParagraphs(docId);
//        System.out.println("ID: " + indexSearcher.doc(docId).get("docid") + "\t#Para: " + para.size());
//        for (int i = 0; i < para.size(); ++i) {
//            System.out.println(para.get(i));
//        }
        //getDocWiseParagraphs();
        //PP1();
        //System.exit(1);
        
        //printVocabularyRaw();
//        //printAllTerms("/store/TCD/TREC_CS/Wor2vec/trunk/64K_BOW_analysed1");
//        printAllTermsFromRaw("/store/TCD/TREC_CS/Wor2vec/trunk/63257_BOW", "/store/TCD/TREC_CS/Wor2vec/trunk/63257_BOW_analysed");
        //printAllTerms("/store/Data/TRECAdhoc/TRECAdhoc_BOW");
        //System.exit(1);
        
//        loadTagsClusters(TRECCSTagsClustersFilePath);
//        loadTagsClustersWeight(TRECCSTagsClustersWeightFilePath);
//        loadW2Vmodel(W2VFilePath); // W2VFilePath // Sample: /store/TCD/TREC_CS/Wor2vec/trunk/a
        loadW2V(W2VFilePath);
        //loadBERT(BERTFilePath);
        

        //getTopDocs();
        //printVocabulary(); // X
        //printAllTermsSelectedDocs();
        //printContext();
        
        // 1. printContextSelected() to get contexts, 2. Use BERT to get context vetors, and 3. mergeContextSelected() to get avg
        //printContextSelected(); // Use this to get contexts of every term in top docs
        //mergeContextSelected(); // After getting BERT (context) vectors of those terms, take mean of multiple (context) vectors of each term
        //System.exit(1);

//        loadBERTDoc(BERTDocFilePath);
//        
//        System.out.println("BERTdoc.size(): " + BERTdoc.size());
//        for (int i = 0; i < 10; ++i) {
//            System.out.println(BERTdoc.get(i).docID);
//            for (int j = 0; j < BERTdoc.get(i).vectors.size(); ++j) {
//                for (int k = 0; k < BERTdoc.get(i).vectors.get(j).length; ++k) {
//                    System.out.print(" " + BERTdoc.get(i).vectors.get(j)[k]);
//                }
//                System.out.println("\n-------------------------------------------------------------");
//            }
//            System.out.println("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
//        }
//        System.exit(1);
        
//        //loadQueryWiseUserPrefOld(queryWiseUserPrefFilePath);
//        //loadQueryWiseUserPref(queryWiseUserPrefFilePath);
        loadQueryWiseUserPref1(queryWiseUserPrefFilePath);                          // TREC-CS req
        loadQueryWiseUserPrefTags(queryWiseUserPrefTagsFilePath);                   // TREC-CS req
        loadQueryWiseUserPrefNegativeTags(queryWiseUserPrefNegativeTagsFilePath);   // TREC-CS req
//        //loadFourSquareData(foursquareDataFilePath);   // Loads Mostafa's Foursquare data
        //loadTECCSQueryJson(treccsQueryJsonFilePath);    // Loads original TRECCS query from .json // TREC-CS req
//        setContextualAppropriateness();                 // Creates 'context', 'category' and 'contextCategoryScore' arrays from Mohammad's crowdsource data

        //readManualQueryVariant();   // Read UQV Robust manual query variants
        
        
        //generateNxNMatrix();  // for cosine similarities between docs for kNN cluster (Lee, Croft & Allan. A cluster-based resampling method for pseudo-relevance feedback. SIGIR '08)
        //getDocDocPairsForNxNMatrixOnUQV();
        //System.exit(1);
        //generateNxNMatrixForkNN_Smart();
        //generateNxNMatrixForManualQVariants();
        //System.exit(1);
        //readNxNMatrix();
        //readNxNMatrix_Smart();

//        writekNNDocs();
//        System.exit(1);

        //readContextualApproTerms();  // Get contextually appropriate terms (classified by SVM). To be used inside KDE.
        
        //initializeRLM(5, 100, 0.4f); // int numFeedbackDocs, int numFeedbackTerms, float QMIX
        //exploreContextualQuery(100, 3, 1000, 1000);
        //exploreContextualQueryGeneral(100, 5, 100, 100);    // int kQueries, int kTerms, int nHits, int nHitsJoint
//        //getParsedQuery();
//        
        

//        for (int i = 0; i < contextualQuery.size(); ++i) {
//            System.out.println(contextualQuery.get(i).context);
////            for (int j = 0; j < contextualQuery.get(i).posTags.length; ++j) {
////                System.out.print(contextualQuery.get(i).posTags[j] + " ");
////            }
////            System.out.println("\n---------------------------------------------------------------");
////            for (int j = 0; j < contextualQuery.get(i).negTags.length; ++j) {
////                System.out.print(contextualQuery.get(i).negTags[j] + " ");
////            }
//            System.out.println("POSITIVE: " + contextualQuery.get(i).posDocs.length);
//            for (int j = 0; j < contextualQuery.get(i).posDocs.length; ++j) {
//                System.out.print(contextualQuery.get(i).posDocs[j] + " ");
//            }
//            System.out.println("\n---------------------------------------------------------------");
//            System.out.println("NEGATIVE: " + contextualQuery.get(i).negDocs.length);
//            for (int j = 0; j < contextualQuery.get(i).negDocs.length; ++j) {
//                System.out.print(contextualQuery.get(i).negDocs[j] + " ");
//            }
//            System.out.println("\n");
//        }

        //analyseResults();
        //System.out.println("SCORE: " + getPoiContextualAppropriatenessSVM("TRECCS-00003071-371", "Group-type:-Friends-AND-Trip-type:-Other-AND-Trip-duration:-Night-out"));
        //System.exit(1);
        
        
//        for (int i = 0; i < foursquareData.size(); ++i) {
//            System.out.print(i + ". " + foursquareData.get(i).TRECId + ": ");
//            for (int j = 0; j < foursquareData.get(i).nCategories; ++j) {
//                System.out.print(foursquareData.get(i).categories.get(j) + " | ");
//            }
//            System.out.println();
//        }
        
//        for (int i = 0; i < treccsQueryJson.size(); ++i) {
//            System.out.println(i + ". " + treccsQueryJson.get(i).qID + ": " + treccsQueryJson.get(i).group + "\t" + treccsQueryJson.get(i).trip_type + "\t" + treccsQueryJson.get(i).duration);
//        }
        
//        for (int i = 0; i < context.length; ++i) {
//            System.out.println(context[i] + "\t" + category[i] + "\tScore: " + contextCategoryScore[i]);
//        }
//        System.out.println("context: " + context.length + "\tcategory: " + category.length + "\tScore: " + contextCategoryScore.length);
        
//        System.out.println("Family and American-Restaurant: " + getContextualAppropriatenessScore("Group_type: Family", "American-Restaurant"));
//        System.out.println("Family and Deli Bodega: " + getContextualAppropriatenessScore("Group_type: Family", "Deli Bodega"));
//        System.out.println("Day trip and States-&-Municipalities: " + getContextualAppropriatenessScore("Trip_duration: Day trip", "States-&-Municipalities"));

        
        //splitTrainTest();
        //getGeoHistogram();
        //getGeoHistogram_1();
        //getGeoHistogram_LTRPairs();
        //System.exit(1);
        
        initializeRLM(10, 20, 0.4f); // int numFeedbackDocs, int numFeedbackTerms, float QMIX
        
//        String[] terms = {"Alone", "Family", "Friends", "Business", "Holiday", "Autumn", "Spring", "Summer"};
//        List<Word2vec> W2V = new ArrayList<>();
//        int termIndex = getTermIndex("beer1");
//        W2V = topkW2Vmodel(termIndex);
//        for (int i = 0; i < 50; ++i) {
//            System.out.println(W2V.get(i).term + " " + W2V.get(i).consineScore);
//        }
//
//        System.out.println(termIndex);
//        System.out.println(indexSearcher.doc(199309).get("docid"));
//        System.out.println("CF: " + getCF("dai"));
//        System.out.println("TF: " + getTF("dai", 164032));
//        System.out.println("IDF: " + getIdf("dai"));
        
        float lambda = 0.5f;
        int nTopTerms = 25;
        
        //docExplore(lambda, nTopTerms);  // Explore here... Selecting top terms to generate queries, without using tags
        
        //System.out.println("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
        //System.exit(1);

        ScoreDoc[] hits = null;

        for (TRECQuery query : queries) {
            
            //query.qtitle = "Art-Galleries Healthy-Food cocktails countryside design fine-dining football history modern-art nature outdoor-activities pizza rafting reef-diving street-Musicians uNESCO-site romantic photography skyline sightseeing local-food pub-hopping amateur-archaeology live-music relaxation Gourmet-Food Museums Organic-Food clean-streets desert fashion-bargains museums art fine-art-museums healthy-food family-friendly deep-sea-fishing Classical-music Live-Music archeology business city-trip country-music diving horse-racing mountaineering nature-walks riverside-walks shopping-for-wine sushi tango wine-travel"; // Most popular K
            
            //System.out.println(query.qid + ": query: " + query.qtitle);
            //TRECQuery queryFiltered = preFilterTRECTags(query, 0.2f); // filter out tags (query terms) that are < 'cutOff' w.r.t. contextual appropriateness (avg.)

//            System.out.print(queryFiltered.qid + ": ");
//            for (int i = 0; i < queryFiltered.nTopTerms; ++i) {
//                //System.out.print(queryFiltered.topTerms.get(i).term + " ");
//                System.out.print(queryFiltered.topTerms.get(i).term + " (" + queryFiltered.topTerms.get(i).weight + ") ");
//            }
//            System.out.println();
            
            //System.out.println(query.qid + ": queryFiltered: " + queryFiltered.qtitle);
            //System.out.println("-----------------------------------------------------");
            //System.exit(1);
            //query.qtitle += " " + getContextualInfo(query);
            
            //query.qtitle = "Art-Galleries Healthy-Food cocktails countryside design fine-dining football history modern-art nature outdoor-activities pizza rafting reef-diving street-Musicians uNESCO-site romantic photography skyline sightseeing local-food pub-hopping amateur-archaeology live-music relaxation Gourmet-Food Museums Organic-Food clean-streets desert fashion-bargains museums art fine-art-museums healthy-food family-friendly deep-sea-fishing Classical-music Live-Music archeology business city-trip country-music diving horse-racing mountaineering nature-walks riverside-walks shopping-for-wine sushi tango wine-travel"; // Most popular K
            
            
            //hits = retrieveGeneral(query, numHits);
            //hits = retrievePOIs(query, numHits);
            //hits = retrieve(query);
            //hits = retrieveCustomized(query, numHits);
            //hits = retrieveCustomized_Geo(query, numHits);
            //hits = retrieveCustomized(queryFiltered, numHits);
            //hits = retrieveCustomized(getExpandedQueryW2VKDE(query, 5, 100), 50);
            //hits = retrieveCustomizedTopTerms(query, numHits, 25);
            //hits = retrieveCustomizedTopTerms(queryFiltered, numHits, 100);
            //hits = retrieveCustomizedTopTerms1(queryFiltered, numHits, 100);
            //hits = CombSUM_TopTerms_And_retrieveCustomized(query, numHits, 25);
            //hits = CombSUM_TopTerms_And_retrieveCustomizedFiltered(queryFiltered, numHits, 25);
            //hits = CombSUM_KDERLM_And_ClassificatTag(query);
            //hits = retrieveMultiQuery(query);
            //hits = retrieveMultiQuery1(query);
            

            //hits = RLMGeneral(query);
            //hits = ariRM(query, numHits);
            //hits = multiRM(query, numHits);
            //hits = RLMGeneralPP(query);
            //hits = RM3Explore(query, 11, 1.0f, 0.8f);
            hits = RM3Explore(query, 8, 0.8f, 0.8f);
            //hits = normalizeMinMax_hits(hits);
            
            //hits = RM3Explore(getExpandedQueryW2VKDE(query, 5, 100), 8, 0.8f, 0.8f);
            
            //hits = naiveBayesExplore(query);
            

                
            //hits = reRankUsingContextualAppropriateness(query, hits);
            //hits = reRankUsingContextualAppropriatenessSVM(query, hits);
            //hits = reRankUsingPOILevelContextualAppropriateness(query, hits);
            //hits = reRankUsingPOILevelContextualAppropriateness_TESTING(query, hits);
            //hits = reRankUsingPOILevelContextualAppropriateness_TESTING_on_prefHistory(query);
            //hits = reRankUsingKLDiv(query, getPOILevelContextualApproDocs(query, numHits), hits);
            //hits = reRankFilteringOnly(query, getPOILevelContextualApproDocs(query, numHits), hits);
            

            //System.exit(1);
            
            if(hits != null) {
                int hits_length = hits.length;
                System.out.println(query.qid + ": documents retrieve: " +hits_length);
                StringBuffer resBuffer = new StringBuffer();

//                System.out.print(query.qid);
//                for (int i = 0; i < 10; ++i) {
//                    if(i < hits.length)
//                        System.out.print(" " + indexSearcher.doc(hits[i].doc).get(FIELD_ID));
//                    else
//                        System.out.print(" Dummy");
//                }
//                System.out.println();
                
                for (int i = 0; i < hits_length; ++i) {
                    int luceneDocId = hits[i].doc;
                    Document d = indexSearcher.doc(luceneDocId);
                    resBuffer.append(query.qid).append("\tQ0\t").
                        append(d.get(FIELD_ID)).append("\t").
                        append((i)).append("\t").
                        append(hits[i].score).append("\t").
                        append(runName).append("\n");
                        //append(runName).append("\t").append(d.get("lat")).append("\n");
                        //append(runName).append("\t").append(query.qlat).append(", ").append(query.qlng).append("\n");
                }
                resFileWriter.write(resBuffer.toString());
            }
        }
        resFileWriter.close();
        System.out.println("The result is saved in: "+resPath);

    }

    public static void main(String[] args) throws IOException, Exception {

        WebDocSearcher_TRECCS_Novel collSearcher = null;

        String usage = "java Wt10gSearcher <properties-file>\n"
            + "Properties file must contain:\n"
            + "1. indexPath: Path of the index\n"
            + "2. fieldToSearch: Name of the field to use for searching\n"
            + "3. queryPath: Path of the query file (in proper xml format)\n"
            + "4. queryFieldFlag: 1-title, 2-title+desc, 3-title+desc+narr\n"
            + "5. similarityFunction: 0.DefaultSimilarity, 1.BM25Similarity, 2.LMJelinekMercerSimilarity, 3.LMDirichletSimilarity\n"
            + "6. param1: \n"
            + "7. [param2]: optional if using BM25";

        /* // uncomment this if wants to run from inside Netbeans IDE
        args = new String[1];
        args[0] = "searcher.properties";
        //*/

        if(0 == args.length) {
            System.out.println(usage);
            System.exit(1);
        }

        System.out.println("Using properties file: "+args[0]);
        collSearcher = new WebDocSearcher_TRECCS_Novel(args[0]);

        collSearcher.retrieveAll();
    }

}
