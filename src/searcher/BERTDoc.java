/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package searcher;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author anirban
 */
public class BERTDoc {
    public String   docID;
    public List<float[]>  vectors = new ArrayList<>(); // Set of paragraph/sentence(context for BERT) vectors for doc 'docID'
}
