"""
Quick Performance Test for RAG App with Caching
"""
import time
import sys
import os

# Add path
sys.path.append('.')

def test_rag_performance():
    """Test RAG system performance with caching"""
    
    print("🚀 RAG Performance Test with Caching")
    print("=" * 50)
    
    try:
        # Import after streamlit setup
        from retriever.parent_retrieval import parent_document_search, generate_answer, initialize_connections
        
        # Initialize connections (first time - should take longer)
        print("🔧 Initializing connections...")
        start = time.time()
        initialize_connections()
        init_time = time.time() - start
        print(f"✅ Initialization: {init_time:.2f}s")
        
        # Test query
        query = "Đảng Cộng sản Việt Nam được thành lập khi nào?"
        print(f"\n🔍 Testing query: {query}")
        
        # First search (should use cached resources but fresh embedding)
        print("\n1️⃣ First search (cached resources + fresh embedding):")
        start = time.time()
        results1 = parent_document_search(query, top_k=3)
        search_time1 = time.time() - start
        print(f"   ⏱️ Search time: {search_time1:.2f}s")
        print(f"   📊 Results found: {len(results1)}")
        
        # Generate answer
        if results1:
            start = time.time()
            context = [(r['parent_chunk']['metadata'], r['parent_id']) for r in results1]
            answer1 = generate_answer(query, context)
            answer_time1 = time.time() - start
            print(f"   ⏱️ Answer generation: {answer_time1:.2f}s")
            total_time1 = search_time1 + answer_time1
            print(f"   🎯 Total time: {total_time1:.2f}s")
        
        # Second search (same query - should use cached embedding)
        print("\n2️⃣ Second search (cached resources + cached embedding):")
        start = time.time()
        results2 = parent_document_search(query, top_k=3)
        search_time2 = time.time() - start
        print(f"   ⏱️ Search time: {search_time2:.2f}s")
        print(f"   📊 Results found: {len(results2)}")
        
        # Generate answer
        if results2:
            start = time.time()
            context = [(r['parent_chunk']['metadata'], r['parent_id']) for r in results2]
            answer2 = generate_answer(query, context)
            answer_time2 = time.time() - start
            print(f"   ⏱️ Answer generation: {answer_time2:.2f}s")
            total_time2 = search_time2 + answer_time2
            print(f"   🎯 Total time: {total_time2:.2f}s")
        
        # Performance comparison
        if results1 and results2:
            print("\n📈 Performance Analysis:")
            search_speedup = search_time1 / search_time2 if search_time2 > 0 else 1
            total_speedup = total_time1 / total_time2 if total_time2 > 0 else 1
            
            print(f"   🔍 Search speedup: {search_speedup:.1f}x faster")
            print(f"   🎯 Total speedup: {total_speedup:.1f}x faster")
            print(f"   ⚡ Time saved: {total_time1 - total_time2:.2f}s ({((total_time1 - total_time2)/total_time1*100):.1f}%)")
        
        # Test different query
        print("\n3️⃣ Different query (cached resources + fresh embedding):")
        new_query = "Tư tưởng Hồ Chí Minh là gì?"
        start = time.time()
        results3 = parent_document_search(new_query, top_k=3)
        search_time3 = time.time() - start
        print(f"   🔍 Query: {new_query}")
        print(f"   ⏱️ Search time: {search_time3:.2f}s")
        print(f"   📊 Results found: {len(results3)}")
        
        print("\n🎉 Performance test completed!")
        print("💡 Expected pattern:")
        print("   - First query: Slower (fresh embedding)")
        print("   - Repeated query: Faster (cached embedding)")
        print("   - New query: Medium (cached resources, fresh embedding)")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        print("💡 Make sure .env file is configured with valid API keys")

if __name__ == "__main__":
    test_rag_performance()
