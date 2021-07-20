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
public class PoiContextualAppropriateness {
    public String   TRECId;
    public String[] jointContext;   // 13 jointContexts
    public double[] score;  // Mohammad's SVM-based scores
}
