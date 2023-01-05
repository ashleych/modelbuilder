from pydantic import BaseModel
from typing import Optional
class  ResultsFeatureselection_internal(BaseModel):
    constant_features:Union[None,List[str]] = None
    quasi_constant_features:Union[None,List[str]] = None
    correlated_features:Union[None,List[str]] = None
    duplicated_features:Union[None,List[str]] = None
    shortlisted_features:Union[None,List[str]] = None
    selected_features:Union[None,List[str]] = None
    cv_scores:Union[None,List[str]] = None
    cv_scores1:Union[None,List[str]] = None
    avg_score:Union[None,List[str]] = None
    rank:Union[None,List[str]] = None
    experiment:Union[None,List[str]] = None

t=ResultsFeatureselection_internal(constant_features=['assd'])