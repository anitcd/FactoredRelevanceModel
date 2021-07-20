/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package searcher;

/**
 *
 * @author anirban
 */
public class DocumentSimilarity {
    public String   q_d_d;  //  queryID, docID1, docID2 ("\t" separated)
    public float    score;  // Cosine similarity between 'docID1' and 'docID2'
}
