# import pickle
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def load_embeddings(filepath="outputs/embeddings.pkl"):
#     print(f"Loading learned embeddings from {filepath}...")
#     with open(filepath, "rb") as f:
#         return pickle.load(f)

# def generate_recommendations(student_idx, embeddings, top_k=3):
#     """
#     Calculates cosine similarity between a target student and all jobs.
#     Returns the top K recommended job indices and their scores.
#     """
#     # 1. Extract embeddings from the PyG dictionary
#     # Assuming PyG output format: {'Student': tensor, 'Job': tensor, ...}
#     student_embs = embeddings["Student"].cpu().detach().numpy()
#     job_embs = embeddings["Job"].cpu().detach().numpy()
    
#     # 2. Get the specific student's vector (reshape for sklearn)
#     target_student_vector = student_embs[student_idx].reshape(1, -1)
    
#     # 3. Calculate Cosine Similarity against all 20 jobs
#     # Returns an array of scores from -1.0 to 1.0
#     similarities = cosine_similarity(target_student_vector, job_embs)[0]
    
#     # 4. Get the indices of the highest scoring jobs
#     best_job_indices = np.argsort(similarities)[::-1][:top_k]
    
#     print(f"\n{'='*60}")
#     print(f" TOP {top_k} JOB RECOMMENDATIONS FOR STUDENT #{student_idx}")
#     print(f"{'='*60}")
    
#     recommendations = []
#     for rank, job_idx in enumerate(best_job_indices):
#         score = similarities[job_idx]
#         recommendations.append((job_idx, score))
#         print(f"Rank {rank + 1}: Job #{job_idx:<3} | GraphSAGE Match Score: {score:.4f}")
        
#     return recommendations

# def check_skill_overlap(student_idx, job_idx, py_graph):
#     """
#     Qualitative Evaluation: Checks if the model's math actually makes sense in reality.
#     Calculates what percentage of the Job's required skills the Student actually has.
#     """
#     # Find all Skill_L3 nodes connected to the Student via KNOWS
#     student_knows_edges = py_graph['Student', 'KNOWS', 'Skill_L3'].edge_index
#     student_skills = set(student_knows_edges[1][student_knows_edges[0] == student_idx].tolist())
    
#     # Find all Skill_L3 nodes connected to the Job via REQUIRES
#     job_requires_edges = py_graph['Job', 'REQUIRES', 'Skill_L3'].edge_index
#     job_skills = set(job_requires_edges[1][job_requires_edges[0] == job_idx].tolist())
    
#     # Prevent division by zero if a job has no skills attached
#     if len(job_skills) == 0:
#         return 0.0, student_skills, job_skills, set()
        
#     # Calculate the overlap
#     overlapping_skills = student_skills.intersection(job_skills)
    
#     # Calculate percentage of required skills met by the student
#     percent_met = len(overlapping_skills) / len(job_skills)
    
#     return percent_met, student_skills, job_skills, overlapping_skills

# if __name__ == "__main__":
#     # 1. Load the embeddings you just trained
#     embs = load_embeddings("outputs/embeddings.pkl")
    
#     # Optional: If you saved your PyG graph during training, load it here for overlap evaluation
#     # py_graph = torch.load("outputs/my_graph_data.pt") 
    
#     # 2. Test the recommendation engine on Student Index 0
#     # You have 9 students, so valid indices are 0 through 8
#     target_student = 0
#     top_jobs = generate_recommendations(student_idx=target_student, embeddings=embs, top_k=3)
    
#     print("\n--- Next Steps for Human Verification ---")
#     print("PyG indices (0, 1, 2...) are different from your Neo4j database IDs.")
#     print("To verify these recommendations are logical, check your raw data:")
#     print(f"1. What are the actual string names of the skills Student #{target_student} has?")
#     for rank, (job_idx, score) in enumerate(top_jobs):
#         print(f"2. What are the string names of the skills required by Job #{job_idx}?")
#     print("-> If the text skills align logically, your GraphSAGE pipeline is a complete success!")

import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(filepath="outputs/embeddings.pkl"):
    print(f"Loading learned embeddings from {filepath}...")
    with open(filepath, "rb") as f:
        return pickle.load(f)

def generate_recommendations(student_idx, embeddings, top_k=3):
    """
    Calculates cosine similarity between a target student and all jobs.
    Converts the raw cosine score into a readable 0-100% probability.
    """
    # 1. Extract embeddings from the PyG dictionary
    student_embs = embeddings["Student"].cpu().detach().numpy()
    job_embs = embeddings["Job"].cpu().detach().numpy()
    
    # 2. Get the specific student's vector (reshape for sklearn)
    target_student_vector = student_embs[student_idx].reshape(1, -1)
    
    # 3. Calculate Cosine Similarity against all jobs
    # Returns an array of scores from -1.0 to 1.0
    similarities = cosine_similarity(target_student_vector, job_embs)[0]
    
    # 4. Sort and get the indices of the highest scoring jobs
    best_job_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n{'='*65}")
    print(f" TOP {top_k} JOB RECOMMENDATIONS FOR STUDENT #{student_idx}")
    print(f"{'='*65}")
    
    recommendations = []
    for rank, job_idx in enumerate(best_job_indices):
        raw_score = similarities[job_idx]
        
        # Translate the -1 to 1 cosine space into a 0% to 100% scale
        match_probability = ((raw_score + 1.0) / 2.0) * 100
        
        recommendations.append((job_idx, match_probability, raw_score))
        print(f"Rank {rank + 1}: Job #{job_idx:<3} | Match: {match_probability:>5.1f}%  (Raw Cosine: {raw_score:>7.4f})")
        
    return recommendations

def check_skill_overlap(student_idx, job_idx, py_graph):
    """
    Qualitative Evaluation: Checks if the model's math actually makes sense in reality.
    Calculates what percentage of the Job's required skills the Student actually has.
    """
    try:
        # Find all Skill_L3 nodes connected to the Student via KNOWS
        student_knows_edges = py_graph['Student', 'KNOWS', 'Skill_L3'].edge_index
        student_skills = set(student_knows_edges[1][student_knows_edges[0] == student_idx].tolist())
        
        # Find all Skill_L3 nodes connected to the Job via REQUIRES
        job_requires_edges = py_graph['Job', 'REQUIRES', 'Skill_L3'].edge_index
        job_skills = set(job_requires_edges[1][job_requires_edges[0] == job_idx].tolist())
        
        # Prevent division by zero if a job has no skills attached
        if len(job_skills) == 0:
            return 0.0, student_skills, job_skills, set()
            
        # Calculate the overlap
        overlapping_skills = student_skills.intersection(job_skills)
        percent_met = (len(overlapping_skills) / len(job_skills)) * 100
        
        return percent_met, student_skills, job_skills, overlapping_skills
        
    except KeyError as e:
        print(f"Warning: Edge type missing from graph - {e}")
        return 0.0, set(), set(), set()

if __name__ == "__main__":
    # 1. Load the embeddings you just trained
    try:
        embs = load_embeddings("outputs/embeddings.pkl")
    except FileNotFoundError:
        print("Error: Could not find outputs/embeddings.pkl. Did you run the training script?")
        exit(1)
    
    # Optional: If you saved your PyG graph during training, load it here for overlap evaluation
    # py_graph = torch.load("outputs/my_graph_data.pt") 
    
    # 2. Test the recommendation engine on Student Index 0
    target_student = 0
    top_jobs = generate_recommendations(student_idx=target_student, embeddings=embs, top_k=3)
    
    print("\n--- Next Steps for Human Verification ---")
    print("PyG indices (0, 1, 2...) are different from your Neo4j database IDs.")
    print("To verify these recommendations are logical, check your raw data:")
    print(f"1. Look at the text of the skills assigned to Student #{target_student}.")
    for rank, (job_idx, match_prob, raw_score) in enumerate(top_jobs):
        print(f"2. Look at the text of the skills required by Rank {rank + 1} Job #{job_idx}.")
    print("-> If they share logical connections, your GraphSAGE pipeline is a complete success!")