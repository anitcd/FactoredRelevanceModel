/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package searcher;

import org.apache.lucene.search.ScoreDoc;

/**
 *
 * @author anirban
 */
public class MultipleRanklists {
    public ScoreDoc[]   hits = null;   // sub-ranklist
    public float        weight = 0.0f; // Weight of the sub-ranklist
    public int          nDocs = 0;  // #docs to consider from the sub-ranklist for the final ranklist
    public String       tagClass = "-1";
}
