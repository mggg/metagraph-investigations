module RecomShadowGraph
open Plotly.NET
open FSharp.Stats
open System.Runtime.Serialization

type plan<'a> = 'a [,]

[<DataContract>]
type ShadowNode = {
    [<DataMember>] String: string
    [<DataMember>] ID: int
    [<DataMember>] Shadow: int [,]
    [<DataMember>] mutable NumberOfPlans: int
    [<DataMember>] mutable AdjacentShadowNodeIDs: int Set
}

type ObservedShadows = Map<string, ShadowNode>


type PlanParser (dim:int) = 
    
    let rec getAdjPairs = function
        | (P: 'a [,], 1)-> Array.concat [|Array.pairwise P.[*, 0]; Array.pairwise P.[0,*];|]
        | (P: 'a [,], length) -> Array.concat [|Array.pairwise P.[*, length-1]; 
                                                Array.pairwise P.[length-1,*]; 
                                                getAdjPairs (P, length-1)|]

    let getShadowPairs P length = getAdjPairs (P, length) 
                                |> Set.ofArray
                                |> Set.filter (fun (a,b) -> a <> b)
                                |> Set.map (fun (a,b) -> Set.ofList [a;b])
                                |> Set.map (fun s -> Set.toList s)

    let InShadow = fun a b -> function
                            | c when c = a -> 0
                            | c when b = c -> 0
                            | _ -> 1

    let ShadowMask P a b = 
        P
        |> Array2D.map (InShadow a b)

    member this.PlanPlot (P: 'a plan) =
        P
        |> JaggedArray.ofArray2D
        |> fun data -> Chart.Heatmap(data, Showscale=false, Colorscale=StyleParam.Colorscale.Portland)
        |> Chart.withSize(400.,400.)

    member this.PlanShadows (P: 'a plan) = 
        getShadowPairs P dim 
        |> Set.toList 
        |> List.map (fun [a;b] -> ShadowMask P a b)

let ShadowPlot (shadow: int [,]) = 
        shadow
        |> JaggedArray.ofArray2D
        |> fun data -> Chart.Heatmap(data, Showscale=false, Colorscale=StyleParam.Colorscale.Portland)
        |> Chart.withSize(400.,400.)

let getPlanShadowNode (shadowNodes: ObservedShadows) (curIndex: int) (shadow: int [,])  = 
    let shadowString = sprintf "%A" shadow
    match Map.tryFind shadowString shadowNodes with
    | Some v -> v, curIndex
    | None -> {String=shadowString; ID=curIndex; Shadow=shadow; NumberOfPlans=0; AdjacentShadowNodeIDs=Set.empty}, curIndex + 1

let updatePlanShadowNode (adjs: int Set) (shadowNodes: ObservedShadows) (shadow: ShadowNode) =
    shadow.NumberOfPlans <- shadow.NumberOfPlans + 1
    shadow.AdjacentShadowNodeIDs <- Set.union shadow.AdjacentShadowNodeIDs adjs
    Map.add shadow.String shadow shadowNodes


let AddNewShadows (shadowNodes: ObservedShadows) (curIndex: int) (planShadows: int [,] list) =
  
    let shadows, newIndex = planShadows |> List.mapFold (getPlanShadowNode shadowNodes) curIndex
    let shadowIndices = shadows |> List.map (fun s -> s.ID) |> Set.ofList

    let newShadowNodes = shadows |> List.fold (updatePlanShadowNode shadowIndices) shadowNodes
    
    newShadowNodes, newIndex


let BuildShadowNodes (plans: 'a plan seq) (parser: PlanParser) = 
    let handlePlan (shadows, index) plan = 
        let planShadows = parser.PlanShadows plan
        AddNewShadows shadows index planShadows

    Seq.fold handlePlan (Map.empty, 0) plans