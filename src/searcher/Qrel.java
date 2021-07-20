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
public class Qrel {
    public String   qID;
    public String   docID;
    public double   relevance;
    
    public double    dist;
    public double    exp;
    public double    relevanceUpdated;
    
    public double    relevanceUpdatedMin;
    public double    relevanceUpdatedMax;
}
