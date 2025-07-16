import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import retriever.cache_data as cache_data       

def parent_document_search(query, namespace="None", p_namespace="None", top_k=10, alpha=0.7):
    """
    Thực hiện Parent Document Retrieval
    
    Args:
        query: Câu hỏi tìm kiếm của người dùng
        namespace: Namespace chứa child chunks trong Pinecone
        top_k: Số lượng parent chunks trả về
        alpha: Trọng số cho việc ranking (0-1)
               alpha = 1: chỉ dựa vào similarity score
               alpha = 0: chỉ dựa vào số lượng child chunks
    
    Returns:
        List các parent chunks được rank theo độ relevance
    """
    dense_index, sparse_index = cache_data.get_pinecone_index()

    # Bước 1: Embed query thành vector
    # print(f"Đang embed query: '{query}'")
    embedding_model = cache_data.get_embedding_model()
    query_vector = embedding_model.encode(query).tolist()
    
    # Bước 2: Tìm kiếm child chunks có score cao
    # Lấy nhiều child chunks để có đủ parent chunks đa dạng
    child_results = dense_index.query(
        vector=query_vector,
        top_k=top_k * 2,  # Lấy gấp 3 lần để đảm bảo có đủ parent unique
        include_metadata=True,
        namespace=namespace
    )
    
    # print(f"Tìm được {len(child_results.matches)} child chunks")
    
    # Bước 3: Gom nhóm child chunks theo parent_id
    parent_scores = {}  # parent_id -> tổng score
    parent_child_count = {}  # parent_id -> số lượng child chunks
    parent_best_score = {}  # parent_id -> score cao nhất
    
    for match in child_results.matches:
        parent_id = match.metadata.get('parent_id')
        
        if parent_id:
            # Khởi tạo nếu chưa có
            if parent_id not in parent_scores:
                parent_scores[parent_id] = 0
                parent_child_count[parent_id] = 0
                parent_best_score[parent_id] = 0
            
            # Cộng dồn score và đếm child chunks
            parent_scores[parent_id] += match.score
            parent_child_count[parent_id] += 1
            parent_best_score[parent_id] = max(parent_best_score[parent_id], match.score)
    
    # print(f"Gom được {len(parent_scores)} parent IDs unique")
    
    # Bước 4: Tính score tổng hợp và rank các parent chunks
    ranked_parents = []
    
    for parent_id, total_score in parent_scores.items():
        # Kiểm tra parent_id có tồn tại trong lookup table không
        avg_score = total_score / parent_child_count[parent_id]
        child_count = parent_child_count[parent_id]
        best_score = parent_best_score[parent_id]
        
        # Normalize child count (giả sử max reasonable là 10 child chunks)
        normalized_child_count = min(child_count / 10.0, 1.0)
        
        # Score cuối = alpha * avg_similarity + (1-alpha) * normalized_child_count
        final_score = alpha * avg_score + (1 - alpha) * normalized_child_count
        
        # Lấy parent chunk từ lookup table
        parent_chunk = dense_index.fetch(
            ids=[parent_id],
            namespace=f"{p_namespace}"
        )
        
        parent_chunk = dense_index.fetch(
            ids=[f"{parent_id}"],
            namespace=f"{p_namespace}"
        )['vectors'][f"{parent_id}"]
                
        # Thêm vào kết quả
        ranked_parents.append({
            'parent_chunk': parent_chunk,
            'parent_id': parent_id,
            'score': final_score,
            'avg_child_score': avg_score,
            'best_child_score': best_score,
            'total_child_score': total_score,
            'child_count': child_count
        })
    
    # Bước 5: Sắp xếp theo score giảm dần và trả về top_k
    ranked_parents.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Ranked {len(ranked_parents)} parent chunks, trả về top {top_k}")
    
    return ranked_parents[:top_k]

if __name__ == "__main__":
    # Test function
    query = "Lịch sử Đảng Cộng sản Việt Nam"
    namespace = "lich-su-dang-children"
    p_namespace = "lich-su-dang"
    top_k = 5
    alpha = 0.7
    
    results = parent_document_search(query, namespace, p_namespace, top_k, alpha)
    for res in results:
        print(f"Parent ID: {res['parent_id']}, Score: {res['score']}, Child Count: {res['child_count']}")