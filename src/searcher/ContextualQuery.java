/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package searcher;

import NaiveBayse.de.daslaboratorium.machinelearning.classifier.Classifier;
import common.TRECQuery;
import java.util.List;

/**
 *
 * @author anirban
 */
public class ContextualQuery {
    public String               context;    // e.g. "Trip-duration:-Night-out"
    public String               city;
    public String[]             posTags;    // tags with contextual appropriateness score > 0.2 e.g. Food, American-Restaurant, Arts-&-Entertainment, ...
    public String[]             negTags;    // tags with contextual appropriateness score <= 0.2 e.g. Food, American-Restaurant, Arts-&-Entertainment, ...
    public String[]             posDocs;    // set of docs (docIDs) for training positive class for 'context'
    public String[]             negDocs;    // set of docs (docIDs) for training negative class for 'context'
    
    public List<TermList>       posTagsWeighted;
    public List<TermList>       negTagsWeighted;
    
    public String[]             queryTerms;
    List<TRECQuery>             randomQueries;  // Random queries (query variants) of original query 'context' ~ qID
    
    Classifier<String, String>  bayes;      // Naive Bayes classifier, to be trained with 'posDocs' and 'negDocs'
}
