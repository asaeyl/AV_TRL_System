"""
Autonomous Vehicle Technology Readiness Level (TRL) Assessment System
"""
import collections
import collections.abc

# Patch
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

from experta import *
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import math
import json
import PyPDF2
import docx
from werkzeug.utils import secure_filename
import openai
import google.generativeai as genai

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY', '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')

# ============================================
# GEMINI LLM INTEGRATION
# ============================================

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDPRgfuaRe4B9adJqyI9jMoBq7HluDdkNo')
AVAILABLE_MODELS = []
WORKING_MODEL = None


def initialize_gemini():
    """Initialize Gemini and find working model"""
    global WORKING_MODEL, AVAILABLE_MODELS
    
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not set")
        return False
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini configured")
        
        # List all available models
        print("\n📋 Checking available models...")
        all_models = genai.list_models()
        
        # Filter for generative models
        for model in all_models:
            # SKIP embedding models
            if 'embedding' in model.name.lower():
                continue
            AVAILABLE_MODELS.append(model.name)
            print(f"   - {model.name}")
        
        # Try models in order of preference
        model_candidates = [
            'models/gemini-2.0-flash',
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash',
            'models/gemini-pro',
            'models/gemini-pro-vision'
        ]
        
        print("\n🔍 Testing model compatibility...")
        
        for model_name in model_candidates:
            try:
                # Test if model works with generateContent
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("test")
                
                WORKING_MODEL = model_name
                print(f"✅ SUCCESS: Using {model_name}")
                return True
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ {model_name} - {error_msg[:60]}")
                continue
        
        # If no model found, try any available model
        if AVAILABLE_MODELS:
            print("\n⚠️ Trying first available model...")
            WORKING_MODEL = AVAILABLE_MODELS[0]
            print(f"✅ Using: {WORKING_MODEL}")
            return True
        
        print("❌ No working model found!")
        return False
        
    except Exception as e:
        print(f"❌ Gemini initialization error: {e}")
        return False

# Initialize on startup
initialize_gemini()


# =======================================
# GEMINI LLM RULE GENERATOR
# =======================================

# =======================================
# FLASK APP CONFIGURATION
# =======================================

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'c77395d7c970862bb7ba1a00cc7ab7833b3f07eb2576e5c960233131f7a43a7c')
CORS(app, resources={r"/*": {"origins": "*"}})

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =======================================
# ADAPTIVE SYSTEM COMPONENTS
# =======================================

class AdaptiveAssessmentResult:
    """Unified result class with confidence and reasoning"""
    def __init__(self, trl, recommendations, confidence, reasoning_type, metadata=None):
        self.trl = trl
        self.recommendations = recommendations
        self.confidence = confidence
        self.reasoning_type = reasoning_type
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            'Results': self.trl,
            'Recommeder': self.recommendations,
            'confidence': self.confidence,
            'reasoning_type': self.reasoning_type,
            'metadata': self.metadata
        }

class EnhancedFuzzyInferenceSystem:
    """Handles multi-level answer assessment with nuanced scoring"""
    
    def __init__(self):
        # Weights based on question importance
        self.dimension_weights = {
            'Infrastructure Readiness': {
                'v2x_maturity': 0.20,
                'infrastructural_adequacy': 0.20,
                'fiveG_advancement': 0.20,
                'shc801_codification': 0.20,
                'ev_infra_progress': 0.20
            },
            'Technology Readiness': {
                'av_technology_progress': 0.40,
                'industrial_roadmap_clarity': 0.30,
                'safety_gaps': 0.30
            },
            'Data & Connectivity Readiness': {
                'connectivity_quality': 0.40,
                'data_governance_maturity': 0.30,
                'regulatory_framework_completeness': 0.30
            },
            'Regulatory & Policy Readiness': {
                'national_vision_strength': 0.25,
                'policy_codification_status': 0.25,
                'pilot_projects_scale': 0.25,
                'national_framework_completeness': 0.25
            },
            'Societal & Market Readiness': {
                'public_acceptance_stage': 0.20,
                'public_readiness_growth': 0.20,
                'trust_building_design': 0.20,
                'outreach_scale': 0.20,
                'use_case_viability': 0.20
            }
        }
        
        # TRL thresholds based on average scores (1-3 scale)
        self.trl_thresholds = {
            'TRL 8-9': 2.7,  # Average score ≥ 2.7
            'TRL 7-8': 2.3,  # Average score ≥ 2.3  
            'TRL 6-7': 2.0,  # Average score ≥ 2.0
            'TRL 5-6': 1.7,  # Average score ≥ 1.7
            'TRL 4-5': 1.3,  # Average score ≥ 1.3
            'TRL 3-4': 1.0   # Average score < 1.3
        }

    def infer(self, dimension_name: str, answers: Dict) -> AdaptiveAssessmentResult:
        """Use fuzzy logic to infer TRL from multi-level patterns"""
        
        if dimension_name not in self.dimension_weights:
            return self._default_fallback(answers)
        
        weights = self.dimension_weights[dimension_name]
        
        # Calculate weighted average score (1-3 scale)
        total_weight = 0
        weighted_score = 0
        answered_questions = 0
        
        for factor, weight in weights.items():
            if factor in answers and answers[factor] is not None:
                weighted_score += answers[factor] * weight
                total_weight += weight
                answered_questions += 1
        
        if total_weight > 0:
            average_score = weighted_score / total_weight
        else:
            average_score = 1.0 # Default to nascent
        
        # Map to TRL based on thresholds
        trl = self._map_score_to_trl(average_score)
        confidence = self._calculate_confidence(average_score, answered_questions, len(weights))
        recommendations = self._generate_contextual_recommendations(dimension_name, answers, average_score)
        
        return AdaptiveAssessmentResult(
            trl=trl,
            recommendations=recommendations,
            confidence=confidence,
            reasoning_type="enhanced_fuzzy",
            metadata={
                "average_score": round(average_score, 2),
                "answered_questions": answered_questions,
                "score_breakdown": answers
            }
        )
    
    def _map_score_to_trl(self, average_score: float) -> str:
        """Map average score (1-3) to TRL based on thresholds"""
        if average_score >= self.trl_thresholds['TRL 8-9']:
            return "TRL 8-9"
        elif average_score >= self.trl_thresholds['TRL 7-8']:
            return "TRL 7-8"
        elif average_score >= self.trl_thresholds['TRL 6-7']:
            return "TRL 6-7"
        elif average_score >= self.trl_thresholds['TRL 5-6']:
            return "TRL 5-6"
        elif average_score >= self.trl_thresholds['TRL 4-5']:
            return "TRL 4-5"
        else:
            return "TRL 3-4"
    
    def _calculate_confidence(self, average_score: float, answered: int, total: int) -> float:
        """Calculate confidence based on answer completeness and consistency"""
        completeness = answered / total if total > 0 else 0
        score_consistency = min(1.0, average_score / 3.0)
        
        # Base confidence on completeness and consistency
        base_confidence = (completeness * 0.6 + score_consistency * 0.4)
        
        # Adjust for extreme scores (very high or very low are often more certain)
        if average_score >= 2.5 or average_score <= 1.5:
            confidence_boost = 0.1
        else:
            confidence_boost = 0.0
            
        return min(0.95, base_confidence + confidence_boost)
    
    def _generate_contextual_recommendations(self, dimension: str, answers: Dict, average_score: float) -> List[str]:
        """Generate specific recommendations based on answer patterns"""
        recommendations = []
        
        # Identify weakest areas (scores ≤ 1.5)
        weak_areas = [factor for factor, score in answers.items() if score is not None and score <= 1.5]
       
        # Identify strongest areas (scores ≥ 2.5) 
        strong_areas = [factor for factor, score in answers.items() if score is not None and score >= 2.5]
        
        # General recommendations based on overall maturity
        if average_score >= 2.5:
            recommendations.append("Leverage mature capabilities for accelerated deployment")
            if weak_areas:
                recommendations.append(f"Address remaining gaps in: {self._format_factors(weak_areas[:2])}")
            recommendations.append("Focus on operational excellence and scaling")
            
        elif average_score >= 1.8:
            recommendations.append("Build on developing capabilities with targeted investments")
            recommendations.append(f"Priority improvements needed in: {self._format_factors(weak_areas[:3])}")
            if strong_areas:
                recommendations.append(f"Utilize strengths in: {self._format_factors(strong_areas[:2])}")
                
        else:
            recommendations.append("Foundational development required across multiple domains")
            recommendations.append(f"Critical focus areas: {self._format_factors(weak_areas[:3])}")
            recommendations.append("Develop phased implementation roadmap")
        
        # Dimension-specific recommendations
        recommendations.extend(self._get_dimension_specific_suggestions(dimension, answers))
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def _format_factors(self, factors: List[str]) -> str:
        """Convert factor names to readable text"""
        name_map = {
            # Infrastructure
            'v2x_maturity': 'V2X technology',
            'infrastructural_adequacy': 'physical infrastructure',
            'fiveG_advancement': '5G deployment',
            'shc801_codification': 'standards adoption',
            'ev_infra_progress': 'EV charging infrastructure',
            # Technology
            'av_technology_progress': 'AV technology development',
            'industrial_roadmap_clarity': 'industrial roadmap',
            'safety_gaps': 'safety frameworks',
            # Data & Connectivity
            'connectivity_quality': 'connectivity infrastructure',
            'data_governance_maturity': 'data governance',
            'regulatory_framework_completeness': 'regulatory framework',
            # Regulatory
            'national_vision_strength': 'national vision',
            'policy_codification_status': 'policy codification',
            'pilot_projects_scale': 'pilot projects',
            'national_framework_completeness': 'national framework',
            # Societal
            'public_acceptance_stage': 'public acceptance',
            'public_readiness_growth': 'public readiness growth',
            'trust_building_design': 'trust building programs',
            'outreach_scale': 'public outreach',
            'use_case_viability': 'use case viability'
        }
        readable_names = [name_map.get(f, f) for f in factors]
        return ', '.join(readable_names)
    
    def _get_dimension_specific_suggestions(self, dimension: str, answers: Dict) -> List[str]:
        """Get dimension-specific recommendations"""
        suggestions = []
        
        if dimension == "Infrastructure Readiness":
            if answers.get('v2x_maturity', 0) <= 1.5:
                suggestions.append("Initiate V2X pilot corridors in urban areas")
            if answers.get('shc801_codification', 0) <= 1.5:
                suggestions.append("Accelerate standards adoption through regulatory channels")
                
        elif dimension == "Technology Readiness":
            if answers.get('safety_gaps', 0) <= 1.5:
                suggestions.append("Strengthen safety case development and validation")
            if answers.get('industrial_roadmap_clarity', 0) <= 1.5:
                suggestions.append("Develop comprehensive industrial collaboration roadmap")
                
        elif dimension == "Data & Connectivity Readiness":
            if answers.get('data_governance_maturity', 0) <= 1.5:
                suggestions.append("Establish data trust frameworks for AV data sharing")
            if answers.get('connectivity_quality', 0) <= 1.5:
                suggestions.append("Enhance network infrastructure for low-latency requirements")
                
        elif dimension == "Regulatory & Policy Readiness":
            if answers.get('policy_codification_status', 0) <= 1.5:
                suggestions.append("Translate policy vision into formal legislation")
            if answers.get('pilot_projects_scale', 0) <= 1.5:
                suggestions.append("Expand pilot programs to demonstrate regulatory approach")
                
        elif dimension == "Societal & Market Readiness":
            if answers.get('public_acceptance_stage', 0) <= 1.5:
                suggestions.append("Launch public education and awareness campaigns")
            if answers.get('trust_building_design', 0) <= 1.5:
                suggestions.append("Design transparent trust-building demonstration programs")
        
        return suggestions

    def _default_fallback(self, answers: Dict) -> AdaptiveAssessmentResult:
        """Default fallback when dimension not recognized"""
        return AdaptiveAssessmentResult(
            trl="TRL 4-5",
            recommendations=["Conduct comprehensive domain analysis"],
            confidence=0.3,
            reasoning_type="default_fallback"
        )

class ConfidenceAggregator:
    """Aggregates multiple reasoning results with confidence weighting"""
    
    def aggregate(self, results: List[AdaptiveAssessmentResult]) -> AdaptiveAssessmentResult:
        """Aggregate multiple results into a single confident assessment"""
        if not results:
            return self._create_default_result()
        
        # Filter by minimum confidence
        valid_results = [r for r in results if r.confidence >= 0.3]
        if not valid_results:
            return self._select_most_confident(results)
        
        # Weighted TRL aggregation
        trl_scores = []
        total_confidence = 0
        
        for result in valid_results:
            numeric_trl = self._trl_to_numeric(result.trl)
            weight = result.confidence
            trl_scores.append(numeric_trl * weight)
            total_confidence += weight
        
        if total_confidence > 0:
            weighted_trl = sum(trl_scores) / total_confidence
        else:
            weighted_trl = self._trl_to_numeric(valid_results[0].trl)
        
        final_trl = self._numeric_to_trl(weighted_trl)
        
        # Merge recommendations
        all_recommendations = []
        seen_recommendations = set()
        
        for result in sorted(valid_results, key=lambda x: x.confidence, reverse=True):
            for rec in result.recommendations:
                rec_hash = rec.lower().strip()
                if rec_hash not in seen_recommendations:
                    seen_recommendations.add(rec_hash)
                    all_recommendations.append(rec)
        
        # Calculate final confidence
        final_confidence = min(0.95, max(r.confidence for r in valid_results) * 0.95)
        
        # Determine reasoning type
        reasoning_types = list(set(r.reasoning_type for r in valid_results))
        if len(reasoning_types) == 1:
            final_reasoning = reasoning_types[0]
        else:
            final_reasoning = "hybrid_aggregation"
        
        return AdaptiveAssessmentResult(
            trl=final_trl,
            recommendations=all_recommendations[:6],
            confidence=final_confidence,
            reasoning_type=final_reasoning,
            metadata={
                "sources_aggregated": len(valid_results),
                "reasoning_components": reasoning_types,
                "aggregation_method": "confidence_weighted"
            }
        )
    
    def _create_default_result(self) -> AdaptiveAssessmentResult:
        """Create default result when no reasoning works"""
        return AdaptiveAssessmentResult(
            trl="TRL 4-5",
            recommendations=[
                "Conduct comprehensive domain analysis",
                "Engage subject matter experts for assessment",
                "Develop strategic implementation roadmap"
            ],
            confidence=0.3,
            reasoning_type="default_fallback"
        )
    
    def _select_most_confident(self, results: List[AdaptiveAssessmentResult]) -> AdaptiveAssessmentResult:
        """Select the result with highest confidence"""
        if not results:
            return self._create_default_result()
        
        best_result = max(results, key=lambda x: x.confidence)
        adjusted_confidence = best_result.confidence * 0.7
        
        return AdaptiveAssessmentResult(
            trl=best_result.trl,
            recommendations=best_result.recommendations,
            confidence=adjusted_confidence,
            reasoning_type=best_result.reasoning_type + "_low_confidence",
            metadata=best_result.metadata
        )
    
    def _trl_to_numeric(self, trl: str) -> float:
        """Convert TRL string to numeric value"""
        mapping = {
            "TRL 8-9": 8.5, "TRL 7-8": 7.5, "TRL 6-7": 6.5,
            "TRL 5-6": 5.5, "TRL 4-5": 4.5, "TRL 3-4": 3.5,
            "TRL 2-3": 2.5, "TRL 1-2": 1.5, "TRL 0": 0.0
        }
        return mapping.get(trl, 5.0)
    
    def _numeric_to_trl(self, numeric: float) -> str:
        """Convert numeric value to TRL string"""
        if numeric >= 8.0: return "TRL 8-9"
        if numeric >= 7.0: return "TRL 7-8"
        if numeric >= 6.0: return "TRL 6-7"
        if numeric >= 5.0: return "TRL 5-6"
        if numeric >= 4.0: return "TRL 4-5"
        if numeric >= 3.0: return "TRL 3-4"
        return "TRL 2-3"

class AdaptiveTRLAssessor:
    """Main orchestrator that never returns no score"""
    
    def __init__(self):
        self.rule_engine = None  # Will be set later
        self.fuzzy_system = EnhancedFuzzyInferenceSystem()
        self.aggregator = ConfidenceAggregator()
    
    def set_rule_engine(self, rule_engine):
        """Set the rule engine after it's defined"""
        self.rule_engine = rule_engine
    
    def assess_dimension(self, dimension_name: str, answers: Dict) -> AdaptiveAssessmentResult:
        """Assess a dimension using multiple reasoning layers"""
        
        # Layer 1: Rule-based reasoning (high confidence if matches)
        rule_result = self._rule_based_assessment(dimension_name, answers)

        if rule_result:
            # Rule matched - return directly 
            return rule_result
        else:
            # Layer 2: Fuzzy inference
            fuzzy_result = self.fuzzy_system.infer(dimension_name, answers)
            return fuzzy_result

    def _rule_based_assessment(self, dimension_name: str, answers: Dict) -> Optional[AdaptiveAssessmentResult]:
        """Try rule-based assessment"""
        if not self.rule_engine:
            return None
        
        # Map dimension to fact class
        fact_class_map = {
            "Infrastructure Readiness": InfrastructureAssessment,
            "Technology Readiness": TechnologyAssessment,
            "Data & Connectivity Readiness": DataConnectivityAssessment,
            "Regulatory & Policy Readiness": RegulatoryAssessment,
            "Societal & Market Readiness": SocietalAssessment
        }
        
        fact_class = fact_class_map.get(dimension_name)
        
        if not fact_class:
            return None
    
        try:
            self.rule_engine.reset_system()
            
            cleaned_answers = {}
            for key, value in answers.items():
                if value is not None:
                    # Ensure value is integer (not float)
                    cleaned_answers[key] = int(value)
        
            # Declare the fact with cleaned answers
            fact = fact_class(**cleaned_answers)
            
            self.rule_engine.declare(fact)
        
            # Run the rule engine
            self.rule_engine.run()

            # Only return result if a SPECIFIC rule matched
            if self.rule_engine.matched and self.rule_engine.results:
                result_dict = self.rule_engine.results
                return AdaptiveAssessmentResult(
                    trl=result_dict.get("Results", "TRL 0"),
                    recommendations=result_dict.get("Recommeder", []),
                    confidence=0.95,
                    reasoning_type="rule_based",
                    metadata={"dimension": dimension_name, "rule_matched": True}
                )
            
        except Exception as e:
            print(f"Rule engine error: {e}")
        
        return None
    
# =======================================
# DATA DEFINITIONS
# =======================================

all_recommendations = []

dimension = {
    "Infrastructure Readiness": [
        "What is the maturity level and deployment stage of Vehicle-to-Everything (V2X) communication technology?",
        "What is the current status of deployment and the adequacy of physical infrastructure required for Autonomous Vehicles (AV)?",
        "To what extent have policies regarding 5G connectivity and dedicated spectrum allocation for Autonomous Vehicles (AV) been developed and implemented?",
        "What is the current extent of formal codification and integration of the SHC 801 standard within national infrastructure regulations?",
        "What is the progress and scale of the Electric Vehicle Infrastructure company (EVIQ) expansion?"
    ],
    "Technology Readiness": [
        "To what extent have Autonomous Vehicles (AV) technologies progressed from laboratory prototypes to implementation in real-world, controlled testing and validation environments?",
        "How comprehensive and transparent are the industrial roadmaps provided by local manufacturing and technology sectors in supporting Autonomous Vehicles (AV) implementation?",
        "To what extent have specific gaps in safety cases been identified, particularly those concerning ISO compliance and the expansion of the Operational Design Domain (ODD)?"
    ],
    "Data & Connectivity Readiness": [
        "To what extent does the nation's connectivity infrastructure, including latency, bandwidth, and coverage, meet the data transmission requirements of Autonomous Vehicles (AV)?",
        "What is the current stage of development of foundational data governance frameworks that address data privacy, security, and ownership for Autonomous Vehicles (AV)?",
        "How comprehensive and legally mature is the regulatory framework governing cross-border data flows and related liability for Autonomous Vehicles (AV)?"
    ],
    "Regulatory & Policy Readiness": [
        "To what extent have regulatory sandboxes matured in fostering and mitigating risks associated with Autonomous Vehicle (AV) innovation?",
        "To what extent have policy visions for Autonomous Vehicles (AVs) been implemented through comprehensive and formally enacted legal frameworks?",
        "What are the scale and strategic scope of ongoing Autonomous Vehicle (AV) pilot projects and initiatives?",
        "What is the current stage of development of a comprehensive national framework that addresses licensing, liability, and insurance for Autonomous Vehicles (AVs)?"
    ],
    "Societal & Market Readiness": [
        "What is the current stage of public understanding and acceptance of Autonomous Vehicles (AV)?",
        "To what degree is there measurable, empirical evidence indicating a positive trend in public readiness and acceptance over time?",
        "To what extent are Autonomous Vehicles (AV) pilot programs systematically designed with respect to structured public engagement and transparency intended to foster trust?",
        "How established and consistent are public outreach initiatives and perception tracking surveys?",
        "How clearly defined and economically viable are near-term AV use cases such as ridesharing and last-mile delivery?"
    ]
}

# ===================================
# KNOWLEDGE BASE - FACT DEFINITIONS
# ===================================

class InfrastructureAssessment(Fact):
    # Numeric scores (1=Nascent, 2=Developing, 3=Mature)
    
    # Infrastructure Readiness
    v2x_maturity = Field(int, default=1)
    infrastructural_adequacy = Field(int, default=1)
    fiveG_advancement = Field(int, default=1)
    shc801_codification = Field(int, default=1)
    ev_infra_progress = Field(int, default=1)

class TechnologyAssessment(Fact):
    # Technology Readiness
    av_technology_progress = Field(int, default=1)
    industrial_roadmap_clarity = Field(int, default=1)
    safety_gaps = Field(int, default=1)

class DataConnectivityAssessment(Fact):  
    # Data & Connectivity Readiness
    connectivity_quality = Field(int, default=1)
    data_governance_maturity = Field(int, default=1)
    regulatory_framework_completeness = Field(int, default=1)

class RegulatoryAssessment(Fact):
    # Regulatory & Policy Readiness
    national_vision_strength = Field(int, default=1)
    policy_codification_status = Field(int, default=1)
    pilot_projects_scale = Field(int, default=1)
    national_framework_completeness = Field(int, default=1)

class SocietalAssessment(Fact):
    # Societal & Market Readiness
    public_acceptance_stage = Field(int, default=1)
    public_readiness_growth = Field(int, default=1)
    trust_building_design = Field(int, default=1)
    outreach_scale = Field(int, default=1)
    use_case_viability = Field(int, default=1)

# =======================================
# GLOBAL VARIABLES
# =======================================

all_dimension_results = []
trl_collector = {}

# Initialize Gemini LLM Generator
# llm_generator = GeminiLLMRuleGenerator(GEMINI_API_KEY)

# ==================================
# EXPERT SYSTEM ENGINE
# ==================================

class TRL_Expert(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.matched = False
        self.results = {"Results": "TRL 0", "Recommeder": []}
    
    def reset_system(self):
        """Reset method"""
        self.matched = False
        self.results = {"Results": "TRL 0", "Recommeder": []}
        self.reset()

    # ==================================
    # INFRASTRUCTURE READINESS RULES
    # ==================================
    
    # RULE INFRA-1 : PERFECT - All mature
    @Rule(
        InfrastructureAssessment(
            v2x_maturity=P(lambda x: x == 3),
            infrastructural_adequacy=P(lambda x: x == 3),
            fiveG_advancement=P(lambda x: x == 3),
            shc801_codification=P(lambda x: x == 3),
            ev_infra_progress=P(lambda x: x == 3)
        )
    )
    def infrastructure_optimal(self):
        """All infrastructure elements at mature stage"""
        recommender = [
            "Accelerate V2X roadside pilot corridors.",
            "Link EVIQ rollout with AV fleet testing to ensure real-world integration.",
            "Establish continuous infrastructure monitoring and optimization."
        ]
        self.matched = True
        self.results = {"Results": "TRL 8-9", "Recommeder": recommender, "dimension": "Infrastructure"}
        self.halt()

    # RULE INFRA-2: MOSTLY MATURE but 801 standards missing 
    @Rule(
        InfrastructureAssessment(
            v2x_maturity=P(lambda x: x == 2),
            infrastructural_adequacy=P(lambda x: x == 2),
            fiveG_advancement=P(lambda x: x == 2),
            shc801_codification=P(lambda x: x == 1), # SHC 801 not implemented
            ev_infra_progress=P(lambda x: x == 2)
         )
    )
    def infrastructure_developing(self):
        """Strong core infrastructure with some gaps"""
        recommender = [
            "Develop interim operational guidelines",
            "Prioritize SHC 801 implementation for standards compliance",
            "Establish certification process for existing infrastructure"
        ]
        self.matched = True
        self.results = {"Results": "TRL 7-8", "Recommeder": recommender, "dimension": "Infrastructure"}
        self.halt()
    
    # RULE INFRA-3: All nascent scenario
    @Rule(
        InfrastructureAssessment(
            v2x_maturity=P(lambda x: x == 1),
            infrastructural_adequacy=P(lambda x: x == 1),
            fiveG_advancement=P(lambda x: x == 1),
            shc801_codification=P(lambda x: x == 1),
            ev_infra_progress=P(lambda x: x == 1)
        )
    )
    def infrastructure_nascent(self):
        """Foundational infrastructure development required"""
        recommender = [
            "Develop comprehensive infrastructure master plan aligned with AV roadmap",
            "Prioritize 5G deployment in urban corridors for pilot zones",
            "Establish infrastructure working group with government and industry",
        ]
        self.matched = True
        self.results = {"Results": "TRL 2-3", "Recommeder": recommender, "dimension": "Infrastructure"}
        self.halt()
    
    # RULE INFRA-4: Strong connectivity but weak physical infrastructure
    @Rule(
        InfrastructureAssessment(
            v2x_maturity=P(lambda x: x == 1),
            infrastructural_adequacy=P(lambda x: x == 1),
            fiveG_advancement=P(lambda x: x == 2),
            shc801_codification=P(lambda x: x == 1),
            ev_infra_progress=P(lambda x: x == 2)
        )
    )
    def infrastructure_connectivity_lead(self):
        """Digital infrastructure ahead of physical infrastructure"""
        recommender = [
            "Leverage strong connectivity for smart infrastructure planning",
            "Accelerate road infrastructure upgrades for AV compatibility",
            "Implement V2X pilot corridors to utilize existing 5G coverage",
            "Coordinate physical and digital infrastructure development timelines"
        ]
        self.matched = True
        self.results = {"Results": "TRL 4-5", "Recommeder": recommender, "dimension": "Infrastructure"}
        self.halt()

    # =====================================
    # TECHNOLOGY READINESS RULES
    # =====================================

    # RULE TECH-1: MOSTLY MATURE but safety gap  
    @Rule(
        TechnologyAssessment(
            av_technology_progress=P(lambda x: x == 3),
            industrial_roadmap_clarity=P(lambda x: x == 3),
            safety_gaps=P(lambda x: x == 1) # with safety gaps
        )
    )
    def technology_safety_gaps(self):
        """Advanced technology with critical safety gaps"""
        recommender = [
            "Enhance safety case documentation and validation",
            "Conduct comprehensive safety audits and validation testing",
            "Establish continuous safety monitoring protocols"
        ]
        self.matched = True
        self.results = {"Results": "TRL 5-6", "Recommeder": recommender, "dimension": "Technology"}
        self.halt()

    # RULE TECH-2: MOSTLY MATURE but No roadmap available  
    @Rule(
        TechnologyAssessment(
            av_technology_progress=P(lambda x: x == 3),
            industrial_roadmap_clarity=P(lambda x: x == 1), # No roadmap available
            safety_gaps=P(lambda x: x == 3) 
        )
    )
    def technology_roadmap_gap(self):
        """Technology progressing but lacking ecosystem coordination"""
        recommender = [
            "Develop comprehensive industrial roadmap with stakeholder engagement",
            "Establish public-private partnerships for ecosystem development",
            "Leverage existing technology maturity for strategic planning"
        ]
        self.matched = True
        self.results = {"Results": "TRL 7-8", "Recommeder": recommender, "dimension": "Technology"}
        self.halt()
    
    # RULE TECH-3: All nascent technology
    @Rule(
        TechnologyAssessment(
            av_technology_progress=P(lambda x: x == 1),
            industrial_roadmap_clarity=P(lambda x: x == 1),
            safety_gaps=P(lambda x: x == 1)
        )
    )
    def technology_nascent(self):
        """Early-stage technology development"""
        recommender = [
            "Establish national AV research and development center",
            "Partner with international technology leaders for knowledge transfer",
            "Develop phased technology adoption roadmap",
            "Invest in foundational research and talent development",
            "Begin safety framework development aligned with ISO 26262"
        ]
        self.matched = True
        self.results = {"Results": "TRL 1-2", "Recommeder": recommender, "dimension": "Technology"}
        self.halt()
    
    # RULE TECH-4: Mixed development (2 mature, 1 developing)
    @Rule(
        TechnologyAssessment(
            av_technology_progress=P(lambda x: x == 2),
            industrial_roadmap_clarity=P(lambda x: x == 2),
            safety_gaps=P(lambda x: x == 2)
        )
    )
    def technology_maturing(self):
        """Technology progressing with manageable gaps"""
        recommender = [
            "Continue safety validation testing with expanded ODD coverage",
            "Strengthen industry collaboration through roadmap execution",
            "Prepare for transition from controlled to operational testing"
        ]
        self.matched = True
        self.results = {"Results": "TRL 6-7", "Recommeder": recommender, "dimension": "Technology"}
        self.halt()

    # =====================================
    # DATA & CONNECTIVITY READINESS RULES
    # =====================================
      
    # RULE DATA-1: PERFECT - All mature  
    @Rule(
        DataConnectivityAssessment(
            connectivity_quality=P(lambda x: x == 3),
            data_governance_maturity=P(lambda x: x == 3),
            regulatory_framework_completeness=P(lambda x: x == 3)
        )
    )
    def data_connectivity_mature(self):
        """Strong data and connectivity foundation"""
        recommender = [
            "Establish National AV Data Trust for fleet data management",
            "Develop annotated AV dataset with local traffic scenarios",
            "Implement real-time data sharing protocols between vehicles and infrastructure"
        ]
        self.matched = True
        self.results = {"Results": "TRL 7-8", "Recommeder": recommender, "dimension": "Data & Connectivity"}
        self.halt()
    
    # RULE DATA-2: MOSTLY MATURE but No Data governance established
    @Rule(
        DataConnectivityAssessment(
            connectivity_quality=P(lambda x: x == 3),
            data_governance_maturity=P(lambda x: x == 1), # Data governance established
            regulatory_framework_completeness=P(lambda x: x == 3)
            )
        )
    def data_connectivity_governance_gap(self):
        """Strong connectivity but lacking data governance frameworks"""
        recommender = [
            "Focus on resolving cross-border data flow regulations",
            "Establish international data sharing agreements with clear accountability",
            "Create data ownership and liability frameworks"
        ]
        self.matched = True
        self.results = {"Results":"TRL 5-6","Recommeder": recommender, "dimension": "Data & Connectivity"}
        self.halt()
    
    # RULE DATA-3: All nascent
    @Rule(
        DataConnectivityAssessment(
            connectivity_quality=P(lambda x: x == 1),
            data_governance_maturity=P(lambda x: x == 1),
            regulatory_framework_completeness=P(lambda x: x == 1)
        )
    )
    def data_connectivity_nascent(self):
        """Foundational data and connectivity development needed"""
        recommender = [
            "Develop national connectivity strategy for AV requirements",
            "Establish data governance working group with stakeholders",
            "Draft preliminary data privacy and security frameworks",
            "Assess connectivity gaps and prioritize coverage expansion"
        ]
        self.matched = True
        self.results = {"Results": "TRL 1-2", "Recommeder": recommender, "dimension": "Data & Connectivity"}
        self.halt()

    # RULE DATA-4: Strong governance but weak connectivity
    @Rule(
        DataConnectivityAssessment(
            connectivity_quality=P(lambda x: x == 1),
            data_governance_maturity=P(lambda x: x == 2),
            regulatory_framework_completeness=P(lambda x: x == 2)
        )
    )
    def data_connectivity_infrastructure_gap(self):
        """Strong policy framework but connectivity limitations"""
        recommender = [
            "Accelerate network infrastructure deployment in priority zones",
            "Leverage existing data governance for infrastructure planning",
            "Implement phased connectivity rollout aligned with AV pilots",
            "Consider edge computing solutions for low-latency requirements"
        ]
        self.matched = True
        self.results = {"Results": "TRL 4-5", "Recommeder": recommender, "dimension": "Data & Connectivity"}
        self.halt()

    # ========================================
    # REGULATORY & POLICY READINESS RULES
    # =======================================
    
    # RULE REG-1: PERFECT - All mature 
    @Rule(
        RegulatoryAssessment(
            national_vision_strength=P(lambda x: x == 3),
            policy_codification_status=P(lambda x: x == 3),
            pilot_projects_scale=P(lambda x: x == 3),
            national_framework_completeness=P(lambda x: x == 3)
        )
    )
    def regulatory_comprehensive(self):
        """Comprehensive regulatory environment with full implementation"""
        recommender = [
            "Enact comprehensive AV legislation specifying liability, certification, and insurance",
            "Align national standards with international regulations",
            "Establish regulatory sandbox for innovation testing"
        ]
        self.matched = True
        self.results = {"Results": "TRL 8-9", "Recommeder": recommender, "dimension": "Regulatory"}
        self.halt()

    # RULE REG-2: lacking implementation  
    @Rule(
        RegulatoryAssessment(
            national_vision_strength=P(lambda x: x == 3),
            policy_codification_status=P(lambda x: x == 3),
            pilot_projects_scale=P(lambda x: x == 1), # No pilots
            national_framework_completeness=P(lambda x: x == 1) # No framework
        )
    )
    def regulatory_visionary(self):
        """Strong vision and codified policy but lacking implementation"""
        recommender = [
            "Launch pilot projects to demonstrate regulatory approach",
            "Establish regulatory sandbox for testing",
            "Develop phased implementation plan for regulatory framework"
        ]
        self.matched = True
        self.results = {"Results": "TRL 5-6", "Recommeder": recommender, "dimension": "Regulatory"}
        self.halt()

    # RULE REG-3: All nascent
    @Rule(
        RegulatoryAssessment(
            national_vision_strength=P(lambda x: x == 1),
            policy_codification_status=P(lambda x: x == 1),
            pilot_projects_scale=P(lambda x: x == 1),
            national_framework_completeness=P(lambda x: x == 1)
        )
    )
    def regulatory_nascent(self):
        """Early-stage regulatory development"""
        recommender = [
            "Establish national AV steering committee with multi-stakeholder representation",
            "Develop vision document aligned with national strategic goals",
            "Conduct regulatory gap analysis against international best practices",
            "Create regulatory sandbox framework for innovation testing"
        ]
        self.matched = True
        self.results = {"Results": "TRL 1-2", "Recommeder": recommender, "dimension": "Regulatory"}
        self.halt()

    # RULE REG-4: Pilots without policy framework
    @Rule(
        RegulatoryAssessment(
            national_vision_strength=P(lambda x: x == 2),
            policy_codification_status=P(lambda x: x == 1),
            pilot_projects_scale=P(lambda x: x == 2),
            national_framework_completeness=P(lambda x: x == 1)
        )
    )
    def regulatory_pilots_ahead(self):
        """Active pilots outpacing policy development"""
        recommender = [
            "URGENT: Accelerate policy codification to support existing pilots",
            "Extract lessons learned from pilots to inform policy development",
            "Establish interim regulatory guidelines for ongoing operations",
            "Create fast-track policy development process leveraging pilot insights"
        ]
        self.matched = True
        self.results = {"Results": "TRL 4-5", "Recommeder": recommender, "dimension": "Regulatory"}
        self.halt()

    # =====================================
    # SOCIETAL & MARKET READINESS RULES
    # ====================================

    # RULE SOC-1: MOSTLY Developing
    @Rule(
        SocietalAssessment(
            public_acceptance_stage=P(lambda x: x == 2),
            public_readiness_growth=P(lambda x: x == 2),
            trust_building_design=P(lambda x: x == 2),
            outreach_scale=P(lambda x: x == 2),
            use_case_viability=P(lambda x: x == 2)
        )
    )
    def societal_developing(self):
        """Growing societal readiness with systematic approach"""
        recommender = [
            "Initiate public trust campaigns with transparent reporting",
            "Launch university AV test rides for generational acceptance",
            "Scale successful pilot programs with community engagement"
        ]
        self.matched = True
        self.results = {"Results": "TRL 5-6", "Recommeder": recommender, "dimension": "Societal"}
        self.halt()

    # RULE SOC-2: Very early stage awareness
    @Rule(
        SocietalAssessment(
            public_acceptance_stage=P(lambda x: x == 2),
            public_readiness_growth=P(lambda x: x == 1),
            trust_building_design=P(lambda x: x == 1),
            outreach_scale=P(lambda x: x == 2),
            use_case_viability=P(lambda x: x == 1)
        )
    )
    def societal_early_stage(self):
        """Early-stage societal readiness with mixed awareness"""
        recommender = [
            "Launch public awareness and education campaigns",
            "Initiate small-scale demonstration projects in controlled environments",
            "Conduct market research to understand public concerns and expectations"
        ]
        self.matched = True
        self.results = {"Results": "TRL 3-4", "Recommeder": recommender, "dimension": "Societal"}
        self.halt()

    # RULE SOC-3: All nascent
    @Rule(
        SocietalAssessment(
            public_acceptance_stage=P(lambda x: x == 1),
            public_readiness_growth=P(lambda x: x == 1),
            trust_building_design=P(lambda x: x == 1),
            outreach_scale=P(lambda x: x == 1),
            use_case_viability=P(lambda x: x == 1)
        )
    )
    def societal_nascent(self):
        """Foundational societal readiness building required"""
        recommender = [
            "Launch national public awareness campaign about AV benefits and safety",
            "Conduct baseline public perception surveys to understand concerns",
            "Design educational programs for schools and universities",
            "Identify and engage early adopter communities for pilot participation",
            "Develop clear use cases relevant to local transportation needs"
        ]
        self.matched = True
        self.results = {"Results": "TRL 1-2", "Recommeder": recommender, "dimension": "Societal"}
        self.halt()

    # RULE SOC-4: All mature
    @Rule(
        SocietalAssessment(
            public_acceptance_stage=P(lambda x: x == 3),
            public_readiness_growth=P(lambda x: x == 3),
            trust_building_design=P(lambda x: x == 3),
            outreach_scale=P(lambda x: x == 3),
            use_case_viability=P(lambda x: x == 3)
        )
    )
    def societal_mature(self):
        """Strong societal readiness and market acceptance"""
        recommender = [
            "Transition from pilot demonstrations to commercial service launches",
            "Expand use cases to broader market segments",
            "Maintain transparent safety reporting to sustain trust",
            "Monitor public sentiment continuously for emerging concerns",
            "Share best practices internationally as societal readiness leader"
        ]
        self.matched = True
        self.results = {"Results": "TRL 7-8", "Recommeder": recommender, "dimension": "Societal"}
        self.halt()

# ================================
# ADAPTIVE SYSTEM INITIALIZATION
# ================================

# Create adaptive assessor and link it with the rule engine
adaptive_assessor = AdaptiveTRLAssessor()
adaptive_assessor.set_rule_engine(TRL_Expert())

# Dimension mappings for frontend to backend conversion
dimension_mappings = {
    "infra": {
        "question_order": [
            'v2x_maturity', 
            'infrastructural_adequacy', 
            'fiveG_advancement', 
            'shc801_codification', 
            'ev_infra_progress'
        ],
        "dimension_name": "Infrastructure Readiness"
    },
    "tech": {
        "question_order": [
            'av_technology_progress',
            'industrial_roadmap_clarity', 
            'safety_gaps'
        ],
        "dimension_name": "Technology Readiness"
    },
    "data": {
        "question_order": [
            'connectivity_quality',
            'data_governance_maturity',
            'regulatory_framework_completeness'
        ],
        "dimension_name": "Data & Connectivity Readiness"
    },
    "reg": {
        "question_order": [
            'national_vision_strength',
            'policy_codification_status',
            'pilot_projects_scale',
            'national_framework_completeness'
        ],
        "dimension_name": "Regulatory & Policy Readiness"
    },
    "soc": {
        "question_order": [
            'public_acceptance_stage',
            'public_readiness_growth', 
            'trust_building_design',
            'outreach_scale',
            'use_case_viability'
        ],
        "dimension_name": "Societal & Market Readiness"
    }
}

# ================================
# TRL CALCULATION FUNCTIONS
# ================================

def trl_to_numeric(trl_str):
    """Convert TRL string to numeric value for calculation"""
    if trl_str == "TRL 8-9":
        return 8.5
    elif trl_str == "TRL 7-8":
        return 7.5
    elif trl_str == "TRL 6-7":
        return 6.5
    elif trl_str == "TRL 5-6":
        return 5.5
    elif trl_str == "TRL 4-5":
        return 4.5
    elif trl_str.startswith("TRL "):
        try:
            return float(trl_str.replace("TRL ", ""))
        except:
            return 5.0 # Default
    return 5.0 # Default for unknown formats

def numeric_to_trl(numeric_val):
    """Convert numeric value back to TRL string"""
    if numeric_val >= 8.0:
        return "TRL 8-9"
    elif numeric_val >= 7.0:
        return "TRL 7-8"
    elif numeric_val >= 6.0:
        return "TRL 6-7"
    elif numeric_val >= 5.0:
        return "TRL 5-6"
    elif numeric_val >= 4.0:
        return "TRL 4-5"
    else:
        return "TRL 3-4"

def calculate_overall_trl(dimension_results):
    """Calculate overall TRL based on all dimension results"""
    if not dimension_results:
        return "TRL 0"
    
    numeric_values = []
    for result in dimension_results:
        if "Results" in result:
            numeric_val = trl_to_numeric(result["Results"])
            numeric_values.append(numeric_val)
    
    if not numeric_values:
        return "TRL 4-5"
    
    average_trl = sum(numeric_values) / len(numeric_values)
    return numeric_to_trl(average_trl)

def get_overall_recommendations(dimension_results):
    """Compile recommendations from all dimensions"""
    all_recommendations = []
    
    for result in dimension_results:
        if "Recommeder" in result and result["Recommeder"]:
            all_recommendations.extend(result["Recommeder"])
    
    return all_recommendations

# ================================
# FILE PROCESSING UTILITIES
# ================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    """Extract text from uploaded file"""
    try:
        ext = filepath.rsplit('.', 1)[1].lower()
        
        if ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif ext == 'pdf':
            text = ""
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif ext == 'docx':
            doc = docx.Document(filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        elif ext == 'doc':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        else:
            return ""
    
    except Exception as e:
        print(f"Error extracting text from file: {e}")
        return ""

# def analyze_text_with_llm(text, framework):
    """Use OpenAI to analyze text and generate rules"""
    if not openai.api_key or not text:
        return {
            'summary': 'AI analysis not available',
            'generatedRules': [],
            'recommendations': ['Upload a detailed document for analysis'],
            'confidence': 0.3
        }
    
    try:
        prompt = f"""Analyze this text for {framework} framework and generate expert system rules.

    Text: {text[:4000]}

    Provide JSON response with:
    {{
        "summary": "Brief analysis summary",
        "overallScore": <float 1.0-3.0>,
        "confidence": <float 0.0-1.0>,
        "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
        "generatedRules": [
            "IF condition THEN action",
            "IF condition THEN action",
            "IF condition THEN action"
        ]
    }}

    Generate at least 3 actionable IF-THEN rules based on the text."""

        response = openai.chat.completions.create( #ChatCompletion
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are an expert in {framework} analysis. Generate specific, actionable rules."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"LLM analysis error: {e}")
        return {
            'summary': 'Analysis completed with basic extraction',
            'overallScore': 2.0,
            'confidence': 0.5,
            'recommendations': ['Review document manually', 'Consult domain experts'],
            'generatedRules': [
                f'IF {framework.lower()}_factor > 2.0 THEN readiness = DEVELOPING',
                f'IF {framework.lower()}_factor < 1.5 THEN action = IMPROVEMENT_NEEDED'
            ]
        }

def analyze_text_with_llm(text, framework):
    """Use Gemini to analyze text and generate rules"""
    
    # Check prerequisites
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set")
        return get_fallback_result(framework)
    
    if not text or len(text.strip()) == 0:
        print("❌ No text provided")
        return get_fallback_result(framework)
    
    if not WORKING_MODEL:
        print("❌ No working Gemini model available")
        return get_fallback_result(framework)
    
    try:
        print(f"\n Analyzing with {WORKING_MODEL}...")
        # Create the prompt
        prompt = f"""Analyze this text for {framework} framework and generate expert system rules.

    Text: {text[:4000]}

    Provide JSON response with:
        {{
            "summary": "Brief analysis summary",
            "overallScore": <float 1.0-3.0>,
            "confidence": <float 0.0-1.0>,
            "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
            "generatedRules": [
                "IF condition THEN action",
                "IF condition THEN action",
                "IF condition THEN action"
            ]
        }}

    Generate at least 3 actionable IF-THEN rules based on the text.
    Return ONLY valid JSON, no markdown or code blocks."""

        # Initialize Gemini model
        model = genai.GenerativeModel(WORKING_MODEL)
        
        # Configure generation parameters
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 1500,
        }
        
        # Generate content using Gemini
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract text from response
        content = response.text.strip()
        
        # Remove markdown code blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        # Parse JSON response
        result = json.loads(content)
        
        # Validate fields
        result.setdefault('summary', 'Analysis completed')
        result.setdefault('overallScore', 2.0)
        result.setdefault('confidence', 0.7)
        result.setdefault('recommendations', [])
        result.setdefault('generatedRules', [])
        
        print(f"✅ Gemini analysis successful")
        return result
        
    except json.JSONDecodeError as e:
        """Handle JSON parsing errors"""
        print(f"❌ JSON parsing error: {e}")
        return get_fallback_result(framework)
    
    except Exception as e:
        """Handle all other errors"""
        print(f"❌ LLM analysis error: {e}")

        # Check if it's a model availability issue
        # if "404" in error_msg or "not found" in error_msg:
        #     print("   💡 Model not found. Reinitializing...")
        #     initialize_gemini()
        error_str = str(e)
        if "404" in error_str or "not found" in error_str:
            print("   💡 Model not found. Reinitializing...")
            initialize_gemini()
        
        return get_fallback_result(framework)

def get_fallback_result(framework):
    """Generate fallback result when LLM fails"""
    
    # Readable fallback rules for each dimension
    fallback_rules_map = {
        'Infrastructure Readiness': [
            'When V2X communication is mature AND 5G coverage complete, THEN Infrastructure ready for deployment = (TRL 8-9)',
            'When V2X pilots active AND infrastructure standards developing, THEN Infrastructure progressing = (TRL 6-7)',
            'When V2X in early stages AND 5G limited, THEN Infrastructure needs foundational work = (TRL 3-4)'
        ],
        'Technology Readiness': [
            'When AV technology mature AND safety validated, THEN Technology ready for commercial deployment = (TRL 8-9)',
            'When AV technology demonstrated AND safety frameworks developing, THEN Technology progressing = (TRL 6-7)',
            'When AV technology in early development, THEN Technology needs further research = (TRL 3-4)'
        ],
        'Data & Connectivity Readiness': [
            'When 5G network complete AND data governance mature, THEN Data systems ready = (TRL 8-9)',
            'When 5G expanding AND data frameworks developing, THEN Data infrastructure progressing = (TRL 6-7)',
            'When connectivity limited AND governance early stage, THEN Data systems need development = (TRL 3-4)'
        ],
        'Regulatory & Policy Readiness': [
            'When regulatory framework complete AND multiple pilot programs approved, THEN Regulatory ready = (TRL 8-9)',
            'When regulations developing AND pilots underway, THEN Regulatory framework progressing = (TRL 6-7)',
            'When regulations nascent AND pilots limited, THEN Regulatory framework needs establishment = (TRL 3-4)'
        ],
        'Societal & Market Readiness': [
            'When societal readiness strong across all dimensions, THEN Market ready for commercial scaling = (TRL 7-8)',
            'When public acceptance growing AND pilot programs demonstrating value, THEN Market readiness developing = (TRL 5-6)',
            'When public awareness limited AND trust concerns high, THEN Societal readiness needs building = (TRL 2-3)'
        ]
    }
    
    # Get rules for this framework, or use generic if not found
    rules = fallback_rules_map.get(framework, [
        f'When {framework.lower()} assessment shows progress across key areas, THEN Readiness advancing = (TRL 6-7)',
        f'When {framework.lower()} factors are partially developed, THEN Readiness moderate = (TRL 5-6)',
        f'When {framework.lower()} factors are in early stages, THEN Readiness needs foundational work = (TRL 3-4)'
    ])
    
    return {
        'summary': f'Analysis completed with fallback rules for {framework}',
        'overallScore': 2.0,
        'confidence': 0.3,
        'recommendations': [
            '⚠️ LLM analysis failed - using fallback rules',
            'Upload a detailed document for AI analysis',
            'Verify Gemini API key is correct'
        ],
        'generatedRules': rules
    }
    


# ================================
# FLASK ROUTES
# ================================

@app.route("/", methods=["GET"])
def index():
    """Serve main application page"""
    return render_template("index2.html", dimension=dimension)

@app.route("/trl_partial_service", methods=["POST"])
def trl_partial_service():
    """Process questionnaire assessment (original endpoint)"""
   
    res = request.get_json().get('data', []) 
    trl_type = request.get_json().get('trl_type', '') 
    print(f"Processing {trl_type} with answers: {res}")
    
    if trl_type not in dimension_mappings:
        return jsonify({
            "results": "TRL 0",
            "recommender": ["Invalid dimension type"],
            "match": False,
            "overall": "TRL 0",
            "overall_num": 0,
            "confidence": 0.1,
            "reasoning_type": "error"
        })
    
    scores = [int(score) for score in res]
    mapping = dimension_mappings[trl_type]
    dimension_name = mapping["dimension_name"]
    question_order = mapping["question_order"]
    
    fact_dict = {}
    for i, score in enumerate(scores):
        if i < len(question_order):
            fact_name = question_order[i]
            fact_dict[fact_name] = score
 
    # Fill missing answers with None (if frontend didn't provide all answers)
    for fact_name in question_order:
        if fact_name not in fact_dict:
            fact_dict[fact_name] = None
    
    print(f"Fact dictionary for {dimension_name}: {fact_dict}")
    
    # Use ADAPTIVE assessment (never returns no score)
    adaptive_result = adaptive_assessor.assess_dimension(dimension_name, fact_dict)
    
    # Store in global collector
    trl_collector[trl_type] = {
        'numeric_trl': adaptive_assessor.aggregator._trl_to_numeric(adaptive_result.trl),
        'confidence': adaptive_result.confidence,
        'result': adaptive_result.to_dict()
    }

    # Calculate overall TRL with confidence weighting
    total_weighted_trl = 0
    total_confidence = 0
    
    for dim_data in trl_collector.values():
        total_weighted_trl += dim_data['numeric_trl'] * dim_data['confidence']
        total_confidence += dim_data['confidence']
    
    if total_confidence > 0:
        overall_numeric = total_weighted_trl / total_confidence
    else:
        overall_numeric = 0
    
    overall_level = numeric_to_trl(overall_numeric)
    
    print(f"TRL Collector: {trl_collector}")
    print(f"Adaptive Result - TRL: {adaptive_result.trl}, Confidence: {adaptive_result.confidence}")
    
    return jsonify({
        "results": adaptive_result.trl,
        "recommender": adaptive_result.recommendations,
        "match": adaptive_result.reasoning_type == "rule_based",
        "overall": overall_level,
        "overall_num": overall_numeric,
        "confidence": adaptive_result.confidence,
        "reasoning_type": adaptive_result.reasoning_type,
        "metadata": adaptive_result.metadata
    })

@app.route("/api/upload-file", methods=["POST"])
def upload_file():
    """Handle file upload and analysis (new endpoint for HTML)"""
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    framework = request.form.get('framework', 'PESTLE')
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"File uploaded: {unique_filename} for {framework} analysis")
        
        # Extract text
        text = extract_text_from_file(filepath)
        
        if not text or len(text) < 100:
            return jsonify({
                "error": "Could not extract sufficient text from file",
                "extractedLength": len(text)
            }), 400
        
        # Analyze with LLM
        analysis_result = analyze_text_with_llm(text, framework)
        
        # Add file metadata
        analysis_result['fileInfo'] = {
            'filename': filename,
            'uploadedFilename': unique_filename,
            'fileSize': os.path.getsize(filepath),
            'textLength': len(text),
            'uploadTime': timestamp,
            'framework': framework
        }
        
        print(f"Document analysis completed: {framework}")
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset_assessment", methods=["POST"])
def reset_assessment():
    """Reset the assessment session"""
    session.clear()
    global trl_collector
    trl_collector = {}
    return jsonify({'success': True})

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":

    app.run(debug=True)
