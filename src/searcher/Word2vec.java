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
public class Word2vec {
    public String   term = "";
    public float[]  vector = new float[200];
    public float    consineScore = 0.0f;
    public float    KDEScore = 0.0f;
    
    public String   qID; // query ID for query wise context embedding (BERT)
}
