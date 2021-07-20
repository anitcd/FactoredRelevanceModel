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
public class ContextualAppropriateness {
    public String   context;    // context e.g. "Trip-duration:-Night-out"
    public String   category;   // category e.g. Food, American-Restaurant, Arts-&-Entertainment, ...
    public float    score;      // contextual appropriateness score
    public int      nAssessors; // no. of assessors judged
}
