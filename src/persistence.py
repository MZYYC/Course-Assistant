#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持久化管理模块 - 企业级版本
负责保存和加载RAG系统的状态、配置、历史记录等
支持批量操作、事务管理、性能监控和高级分析
"""

import json
import pickle
import sqlite3
import threading
import time
import zlib
import gzip
import fnmatch
import shutil
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Generator
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config


class PersistenceManager:
    """持久化管理器 - 企业级版本"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化持久化管理器
        
        Args:
            storage_path: 存储路径，默认使用配置中的路径
        """
        self.storage_path = Path(storage_path or Config.VECTOR_STORE_PATH).parent / "persistence"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 数据库文件路径
        self.db_path = self.storage_path / "rag_system.db"
        self.config_path = self.storage_path / "system_config.json"
        self.cache_path = self.storage_path / "cache"
        self.cache_path.mkdir(exist_ok=True)
        self.vector_store_path = Path(Config.VECTOR_STORE_PATH)
        
        # 企业级功能初始化
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._max_connections = 10
        self._performance_stats = {
            'operations_count': {},
            'operation_times': {},
            'last_cleanup': None,
            'db_size_history': [],
            'cache_hit_rate': {'hits': 0, 'misses': 0},
            'initialization_time': datetime.now().isoformat()
        }
        
        # 缓存锁和元数据
        self._cache_lock = threading.Lock()
        self._cache_metadata = {}
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="persistence")
        
        # 设置日志
        self._setup_logging()
        
        # 初始化数据库
        self._init_database()
        
        # 初始化性能监控
        self._init_performance_monitoring()
        
        print(f"持久化管理器初始化完成 (企业级版本)，存储路径: {self.storage_path}")
    
    def _setup_logging(self):
        """设置日志记录"""
        log_path = self.storage_path / "persistence.log"
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_performance_monitoring(self):
        """初始化性能监控"""
        self._performance_stats['initialization_time'] = datetime.now().isoformat()
        self._record_operation('init', 0.0)  # 占位记录
    
    def _record_operation(self, operation: str, duration: float):
        """记录操作性能"""
        if operation not in self._performance_stats['operations_count']:
            self._performance_stats['operations_count'][operation] = 0
            self._performance_stats['operation_times'][operation] = []
        
        self._performance_stats['operations_count'][operation] += 1
        self._performance_stats['operation_times'][operation].append(duration)
        
        # 保留最近100次操作的时间记录
        if len(self._performance_stats['operation_times'][operation]) > 100:
            self._performance_stats['operation_times'][operation] = \
                self._performance_stats['operation_times'][operation][-100:]
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """获取数据库连接（连接池管理）"""
        conn = None
        try:
            with self._pool_lock:
                if self._connection_pool:
                    conn = self._connection_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            
            yield conn
            
        except Exception as e:
            self.logger.error(f"数据库连接错误: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                with self._pool_lock:
                    if len(self._connection_pool) < self._max_connections:
                        self._connection_pool.append(conn)
                    else:
                        conn.close()
    
    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """事务管理上下文"""
        with self._get_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"事务回滚: {e}")
                raise
    
    def _initialize_connection_pool(self):
        """初始化连接池"""
        with self._pool_lock:
            # 关闭现有连接
            for conn in self._connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self._connection_pool.clear()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        start_time = time.time()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 查询历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT,
                    sources TEXT,
                    strategy TEXT,
                    response_time REAL,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    user_session TEXT,
                    query_hash TEXT
                )
            """)
            
            # 文档处理历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER,
                    processing_time REAL,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    chunk_count INTEGER,
                    strategy TEXT,
                    file_hash TEXT,
                    metadata TEXT
                )
            """)
            
            # 系统性能统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    tags TEXT
                )
            """)
            
            # 用户会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    query_count INTEGER DEFAULT 0,
                    last_activity TEXT,
                    user_agent TEXT,
                    ip_address TEXT
                )
            """)
            
            # 批量操作记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_operations (
                    batch_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT DEFAULT 'running',
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_strategy ON query_history(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_timestamp ON document_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_strategy ON document_history(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON system_stats(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_category ON system_stats(category)")
            
            conn.commit()
        
        operation_time = time.time() - start_time
        self._record_operation('init_database', operation_time)
        self.logger.info(f"数据库初始化完成，耗时: {operation_time:.4f}s")
    
    # ==================== 查询历史管理 ====================
    
    def save_query_history(
        self, 
        query: str, 
        answer: Optional[str] = None, 
        sources: Optional[List[Dict]] = None,
        strategy: Optional[str] = None,
        response_time: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        user_session: Optional[str] = None
    ) -> str:
        """
        保存查询历史
        
        Returns:
            query_id: 查询记录ID
        """
        start_time = time.time()
        
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        query_hash = str(hash(query))
        
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_history 
                    (id, timestamp, query, answer, sources, strategy, response_time, 
                     success, error_message, user_session, query_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (query_id, timestamp, query, answer, sources_json, strategy, 
                      response_time, success, error_message, user_session, query_hash))
            
            operation_time = time.time() - start_time
            self._record_operation('save_query_history', operation_time)
            
            return query_id
            
        except Exception as e:
            self.logger.error(f"保存查询历史失败: {e}")
            raise
    
    def save_query_history_batch(self, queries: List[Dict]) -> List[str]:
        """批量保存查询历史"""
        start_time = time.time()
        query_ids = []
        
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                
                for query_data in queries:
                    query_id = str(uuid.uuid4())
                    timestamp = datetime.now().isoformat()
                    
                    # 处理源数据
                    sources_json = None
                    if query_data.get('sources'):
                        sources_json = json.dumps(query_data['sources'], ensure_ascii=False)
                    
                    query_hash = str(hash(query_data['query']))
                    
                    cursor.execute("""
                        INSERT INTO query_history 
                        (id, timestamp, query, answer, sources, strategy, response_time, 
                         success, error_message, user_session, query_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        query_id, timestamp, query_data['query'], 
                        query_data.get('answer'), sources_json,
                        query_data.get('strategy'), query_data.get('response_time'),
                        query_data.get('success', True), query_data.get('error_message'),
                        query_data.get('user_session'), query_hash
                    ))
                    
                    query_ids.append(query_id)
            
            operation_time = time.time() - start_time
            self._record_operation('save_query_history_batch', operation_time)
            self.logger.info(f"批量保存 {len(queries)} 条查询历史，耗时: {operation_time:.4f}s")
            
            return query_ids
            
        except Exception as e:
            self.logger.error(f"批量保存查询历史失败: {e}")
            raise
    
    def get_query_history(
        self, 
        limit: int = 100, 
        success_only: bool = False,
        strategy: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_session: Optional[str] = None
    ) -> List[Dict]:
        """获取查询历史（增强版）"""
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                sql = "SELECT * FROM query_history"
                params = []
                conditions = []
                
                if success_only:
                    conditions.append("success = 1")
                if strategy:
                    conditions.append("strategy = ?")
                    params.append(strategy)
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(start_date)
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(end_date)
                if user_session:
                    conditions.append("user_session = ?")
                    params.append(user_session)
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                columns = [desc[0] for desc in cursor.description]
                
                results = []
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    if record['sources']:
                        try:
                            record['sources'] = json.loads(record['sources'])
                        except:
                            record['sources'] = []
                    results.append(record)
                
                operation_time = time.time() - start_time
                self._record_operation('get_query_history', operation_time)
                
                return results
                
        except Exception as e:
            self.logger.error(f"获取查询历史失败: {e}")
            raise
    
    def get_query_statistics(self, advanced: bool = False) -> Dict[str, Any]:
        """获取查询统计信息（增强版）"""
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 基础统计
                cursor.execute("SELECT COUNT(*) FROM query_history")
                total_queries = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM query_history WHERE success = 1")
                successful_queries = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(response_time) FROM query_history WHERE success = 1 AND response_time IS NOT NULL")
                avg_response_time = cursor.fetchone()[0] or 0
                
                # 按策略统计
                cursor.execute("""
                    SELECT strategy, COUNT(*) as count, AVG(response_time) as avg_time,
                           MIN(response_time) as min_time, MAX(response_time) as max_time
                    FROM query_history 
                    WHERE success = 1 AND strategy IS NOT NULL AND response_time IS NOT NULL
                    GROUP BY strategy
                """)
                strategy_stats = {}
                for row in cursor.fetchall():
                    strategy_stats[row[0]] = {
                        'count': row[1],
                        'avg_response_time': row[2] or 0,
                        'min_response_time': row[3] or 0,
                        'max_response_time': row[4] or 0
                    }
                
                # 时间段统计
                time_periods = {
                    'last_hour': 1,
                    'last_24h': 24,
                    'last_week': 24 * 7,
                    'last_month': 24 * 30
                }
                
                time_stats = {}
                for period, hours in time_periods.items():
                    cursor.execute("""
                        SELECT COUNT(*) FROM query_history 
                        WHERE datetime(timestamp) > datetime('now', '-{} hours')
                    """.format(hours))
                    time_stats[period] = cursor.fetchone()[0]
                
                basic_stats = {
                    'total_queries': total_queries,
                    'successful_queries': successful_queries,
                    'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
                    'avg_response_time': avg_response_time,
                    'strategy_stats': strategy_stats,
                    'time_period_stats': time_stats
                }
                
                if advanced:
                    # 高级统计分析
                    advanced_stats = self._get_advanced_query_analytics(cursor)
                    basic_stats.update(advanced_stats)
                
                operation_time = time.time() - start_time
                self._record_operation('get_query_statistics', operation_time)
                
                return basic_stats
                
        except Exception as e:
            self.logger.error(f"获取查询统计失败: {e}")
            raise
    
    def _get_advanced_query_analytics(self, cursor: sqlite3.Cursor) -> Dict[str, Any]:
        """获取高级查询分析"""
        advanced_stats = {}
        
        # 查询频率分析
        cursor.execute("""
            SELECT query_hash, COUNT(*) as frequency
            FROM query_history 
            GROUP BY query_hash 
            ORDER BY frequency DESC 
            LIMIT 10
        """)
        top_queries = [{'hash': row[0], 'frequency': row[1]} for row in cursor.fetchall()]
        advanced_stats['top_queries_by_frequency'] = top_queries
        
        # 性能分布分析
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN response_time < 1.0 THEN '< 1s'
                    WHEN response_time < 3.0 THEN '1-3s'
                    WHEN response_time < 5.0 THEN '3-5s'
                    WHEN response_time < 10.0 THEN '5-10s'
                    ELSE '> 10s'
                END as time_range,
                COUNT(*) as count
            FROM query_history 
            WHERE response_time IS NOT NULL
            GROUP BY time_range
        """)
        performance_distribution = dict(cursor.fetchall())
        advanced_stats['performance_distribution'] = performance_distribution
        
        # 错误分析
        cursor.execute("""
            SELECT error_message, COUNT(*) as count
            FROM query_history 
            WHERE success = 0 AND error_message IS NOT NULL
            GROUP BY error_message
            ORDER BY count DESC
            LIMIT 5
        """)
        error_analysis = [{'error': row[0], 'count': row[1]} for row in cursor.fetchall()]
        advanced_stats['top_errors'] = error_analysis
        
        # 会话分析
        cursor.execute("""
            SELECT user_session, COUNT(*) as query_count, 
                   AVG(response_time) as avg_response_time
            FROM query_history 
            WHERE user_session IS NOT NULL
            GROUP BY user_session
            HAVING query_count > 1
            ORDER BY query_count DESC
            LIMIT 10
        """)
        session_analysis = []
        for row in cursor.fetchall():
            session_analysis.append({
                'session': row[0],
                'query_count': row[1],
                'avg_response_time': row[2] or 0
            })
        advanced_stats['top_sessions'] = session_analysis
        
        return advanced_stats
    
    # ==================== 文档处理历史 ====================
    
    def save_document_history(
        self,
        file_path: str,
        file_name: str,
        file_size: Optional[int] = None,
        processing_time: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        strategy: Optional[str] = None,
        file_hash: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """保存文档处理历史（增强版）"""
        start_time = time.time()
        
        doc_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO document_history 
                    (id, timestamp, file_path, file_name, file_size, processing_time, 
                     success, error_message, chunk_count, strategy, file_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (doc_id, timestamp, file_path, file_name, file_size, 
                      processing_time, success, error_message, chunk_count, 
                      strategy, file_hash, metadata_json))
            
            operation_time = time.time() - start_time
            self._record_operation('save_document_history', operation_time)
            
            return doc_id
            
        except Exception as e:
            self.logger.error(f"保存文档历史失败: {e}")
            raise
    
    def save_document_history_batch(self, documents: List[Dict]) -> List[str]:
        """批量保存文档处理历史"""
        start_time = time.time()
        doc_ids = []
        batch_id = str(uuid.uuid4())
        
        try:
            # 记录批量操作开始
            self._record_batch_operation_start(batch_id, 'document_history_batch', len(documents))
            
            with self._transaction() as conn:
                cursor = conn.cursor()
                
                for doc_data in documents:
                    doc_id = str(uuid.uuid4())
                    timestamp = datetime.now().isoformat()
                    
                    metadata_json = None
                    if doc_data.get('metadata'):
                        metadata_json = json.dumps(doc_data['metadata'], ensure_ascii=False)
                    
                    cursor.execute("""
                        INSERT INTO document_history 
                        (id, timestamp, file_path, file_name, file_size, processing_time, 
                         success, error_message, chunk_count, strategy, file_hash, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc_id, timestamp, doc_data['file_path'], doc_data['file_name'],
                        doc_data.get('file_size'), doc_data.get('processing_time'),
                        doc_data.get('success', True), doc_data.get('error_message'),
                        doc_data.get('chunk_count'), doc_data.get('strategy'),
                        doc_data.get('file_hash'), metadata_json
                    ))
                    
                    doc_ids.append(doc_id)
            
            # 记录批量操作完成
            operation_time = time.time() - start_time
            self._record_batch_operation_end(batch_id, 'completed', len(doc_ids), 0)
            self._record_operation('save_document_history_batch', operation_time)
            self.logger.info(f"批量保存 {len(documents)} 条文档历史，耗时: {operation_time:.4f}s")
            
            return doc_ids
            
        except Exception as e:
            self._record_batch_operation_end(batch_id, 'failed', len(doc_ids), 1)
            self.logger.error(f"批量保存文档历史失败: {e}")
            raise
    
    def _record_batch_operation_start(self, batch_id: str, operation_type: str, total_items: int):
        """记录批量操作开始"""
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO batch_operations 
                    (batch_id, operation_type, start_time, total_items)
                    VALUES (?, ?, ?, ?)
                """, (batch_id, operation_type, datetime.now().isoformat(), total_items))
        except Exception as e:
            self.logger.error(f"记录批量操作开始失败: {e}")
    
    def _record_batch_operation_end(self, batch_id: str, status: str, processed_items: int, error_count: int):
        """记录批量操作结束"""
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE batch_operations 
                    SET end_time = ?, status = ?, processed_items = ?, error_count = ?
                    WHERE batch_id = ?
                """, (datetime.now().isoformat(), status, processed_items, error_count, batch_id))
        except Exception as e:
            self.logger.error(f"记录批量操作结束失败: {e}")
    
    def get_document_history(
        self, 
        limit: int = 100,
        success_only: bool = False,
        strategy: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """获取文档处理历史（增强版）"""
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                sql = "SELECT * FROM document_history"
                params = []
                conditions = []
                
                if success_only:
                    conditions.append("success = 1")
                if strategy:
                    conditions.append("strategy = ?")
                    params.append(strategy)
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(start_date)
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(end_date)
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                columns = [desc[0] for desc in cursor.description]
                
                results = []
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    if record['metadata']:
                        try:
                            record['metadata'] = json.loads(record['metadata'])
                        except:
                            record['metadata'] = {}
                    results.append(record)
                
                operation_time = time.time() - start_time
                self._record_operation('get_document_history', operation_time)
                
                return results
                
        except Exception as e:
            self.logger.error(f"获取文档历史失败: {e}")
            raise
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """获取文档处理统计信息"""
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 基础统计
                cursor.execute("SELECT COUNT(*) FROM document_history")
                total_documents = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM document_history WHERE success = 1")
                successful_documents = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(processing_time), SUM(file_size), AVG(chunk_count) FROM document_history WHERE success = 1")
                result = cursor.fetchone()
                avg_processing_time = result[0] or 0
                total_file_size = result[1] or 0
                avg_chunk_count = result[2] or 0
                
                # 按策略统计
                cursor.execute("""
                    SELECT strategy, COUNT(*) as count, AVG(processing_time) as avg_time,
                           SUM(file_size) as total_size, AVG(chunk_count) as avg_chunks
                    FROM document_history 
                    WHERE success = 1 AND strategy IS NOT NULL
                    GROUP BY strategy
                """)
                strategy_stats = {}
                for row in cursor.fetchall():
                    strategy_stats[row[0]] = {
                        'count': row[1],
                        'avg_processing_time': row[2] or 0,
                        'total_file_size': row[3] or 0,
                        'avg_chunk_count': row[4] or 0
                    }
                
                # 文件类型统计
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN file_name LIKE '%.pdf' THEN 'PDF'
                            WHEN file_name LIKE '%.txt' THEN 'TXT'
                            WHEN file_name LIKE '%.docx' THEN 'DOCX'
                            WHEN file_name LIKE '%.md' THEN 'Markdown'
                            ELSE 'Other'
                        END as file_type,
                        COUNT(*) as count
                    FROM document_history
                    GROUP BY file_type
                """)
                file_type_stats = dict(cursor.fetchall())
                
                operation_time = time.time() - start_time
                self._record_operation('get_document_statistics', operation_time)
                
                return {
                    'total_documents': total_documents,
                    'successful_documents': successful_documents,
                    'success_rate': successful_documents / total_documents if total_documents > 0 else 0,
                    'avg_processing_time': avg_processing_time,
                    'total_file_size': total_file_size,
                    'avg_chunk_count': avg_chunk_count,
                    'strategy_stats': strategy_stats,
                    'file_type_stats': file_type_stats
                }
                
        except Exception as e:
            self.logger.error(f"获取文档统计失败: {e}")
            raise
    
    # ==================== 系统配置持久化 ====================
    
    def save_system_config(self, config_data: Dict[str, Any]):
        """保存系统配置（增强版）"""
        start_time = time.time()
        
        try:
            config_data['last_updated'] = datetime.now().isoformat()
            config_data['version'] = getattr(config_data, 'version', '1.0.0')
            
            # 备份现有配置
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.backup.json')
                self.config_path.rename(backup_path)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            operation_time = time.time() - start_time
            self._record_operation('save_system_config', operation_time)
            self.logger.info(f"系统配置已保存到: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"保存系统配置失败: {e}")
            raise
    
    def load_system_config(self) -> Dict[str, Any]:
        """加载系统配置（增强版）"""
        start_time = time.time()
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                operation_time = time.time() - start_time
                self._record_operation('load_system_config', operation_time)
                self._performance_stats['cache_hit_rate']['hits'] += 1
                
                return config
            else:
                self._performance_stats['cache_hit_rate']['misses'] += 1
                return {}
                
        except Exception as e:
            self.logger.error(f"加载系统配置时出错: {e}")
            self._performance_stats['cache_hit_rate']['misses'] += 1
            return {}
    
    # ==================== 缓存管理 ====================
    
    def save_cache(self, cache_key: str, data: Any, expire_hours: int = 24, compress: bool = False):
        """保存缓存数据（增强版）"""
        start_time = time.time()
        
        try:
            cache_file = self.cache_path / f"{cache_key}.pkl"
            cache_data = {
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'expire_hours': expire_hours,
                'compressed': compress,
                'size': len(str(data)) if not compress else 0
            }
            
            # 可选压缩
            if compress and len(str(data)) > 1024:  # 大于1KB时压缩
                cache_file = self.cache_path / f"{cache_key}.gz.pkl"
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            operation_time = time.time() - start_time
            self._record_operation('save_cache', operation_time)
            
        except Exception as e:
            self.logger.error(f"保存缓存失败: {e}")
            raise
    
    def load_cache(self, cache_key: str) -> Optional[Any]:
        """加载缓存数据（增强版）"""
        start_time = time.time()
        
        # 尝试加载压缩和非压缩版本
        cache_files = [
            self.cache_path / f"{cache_key}.gz.pkl",
            self.cache_path / f"{cache_key}.pkl"
        ]
        
        for cache_file in cache_files:
            if not cache_file.exists():
                continue
                
            try:
                if cache_file.name.endswith('.gz.pkl'):
                    with gzip.open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                else:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                
                # 检查是否过期
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                expire_hours = cache_data.get('expire_hours', 24)
                
                if (datetime.now() - cache_time).total_seconds() > expire_hours * 3600:
                    cache_file.unlink()  # 删除过期缓存
                    self._performance_stats['cache_hit_rate']['misses'] += 1
                    continue
                
                operation_time = time.time() - start_time
                self._record_operation('load_cache', operation_time)
                self._performance_stats['cache_hit_rate']['hits'] += 1
                
                return cache_data['data']
                
            except Exception as e:
                self.logger.error(f"加载缓存 {cache_key} 时出错: {e}")
                cache_file.unlink()  # 删除损坏的缓存
                continue
        
        self._performance_stats['cache_hit_rate']['misses'] += 1
        return None
    
    def clear_cache(self, cache_key: Optional[str] = None, pattern: Optional[str] = None):
        """清除缓存（增强版）"""
        start_time = time.time()
        cleared_count = 0
        
        try:
            if cache_key:
                # 清除特定缓存
                cache_files = [
                    self.cache_path / f"{cache_key}.pkl",
                    self.cache_path / f"{cache_key}.gz.pkl"
                ]
                for cache_file in cache_files:
                    if cache_file.exists():
                        cache_file.unlink()
                        cleared_count += 1
                        
            elif pattern:
                # 按模式清除
                for cache_file in self.cache_path.glob("*.pkl"):
                    if fnmatch.fnmatch(cache_file.stem, pattern):
                        cache_file.unlink()
                        cleared_count += 1
                        
            else:
                # 清除所有缓存
                for cache_file in self.cache_path.glob("*.pkl"):
                    cache_file.unlink()
                    cleared_count += 1
                for cache_file in self.cache_path.glob("*.gz.pkl"):
                    cache_file.unlink()
                    cleared_count += 1
            
            operation_time = time.time() - start_time
            self._record_operation('clear_cache', operation_time)
            self.logger.info(f"清除了 {cleared_count} 个缓存文件")
            
        except Exception as e:
            self.logger.error(f"清除缓存失败: {e}")
            raise
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            cache_files = list(self.cache_path.glob("*.pkl")) + list(self.cache_path.glob("*.gz.pkl"))
            
            total_size = sum(f.stat().st_size for f in cache_files)
            total_files = len(cache_files)
            
            hit_rate = 0
            if self._performance_stats['cache_hit_rate']['hits'] + self._performance_stats['cache_hit_rate']['misses'] > 0:
                hit_rate = self._performance_stats['cache_hit_rate']['hits'] / (
                    self._performance_stats['cache_hit_rate']['hits'] + 
                    self._performance_stats['cache_hit_rate']['misses']
                )
            
            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self._performance_stats['cache_hit_rate']['hits'],
                'misses': self._performance_stats['cache_hit_rate']['misses']
            }
            
        except Exception as e:
            self.logger.error(f"获取缓存统计失败: {e}")
            return {}
    
    # ==================== 系统状态持久化 ====================
    
    def save_system_state(self, state_data: Dict[str, Any]):
        """保存系统状态（增强版）"""
        start_time = time.time()
        
        try:
            state_file = self.storage_path / "system_state.json"
            state_data['timestamp'] = datetime.now().isoformat()
            state_data['performance_stats'] = self._performance_stats
            
            # 添加系统健康状态
            state_data['health_status'] = self._get_system_health()
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
            
            operation_time = time.time() - start_time
            self._record_operation('save_system_state', operation_time)
            
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {e}")
            raise
    
    def load_system_state(self) -> Dict[str, Any]:
        """加载系统状态（增强版）"""
        start_time = time.time()
        
        try:
            state_file = self.storage_path / "system_state.json"
            
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                operation_time = time.time() - start_time
                self._record_operation('load_system_state', operation_time)
                
                return state
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"加载系统状态时出错: {e}")
            return {}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 数据库大小
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # 缓存目录大小
            cache_size = sum(f.stat().st_size for f in self.cache_path.glob("*") if f.is_file())
            
            # 连接池状态
            pool_status = {
                'active_connections': len(self._connection_pool),
                'max_connections': self._max_connections
            }
            
            # 性能指标
            avg_operation_times = {}
            for op, times in self._performance_stats['operation_times'].items():
                if times:
                    avg_operation_times[op] = sum(times) / len(times)
            
            return {
                'db_size_bytes': db_size,
                'cache_size_bytes': cache_size,
                'connection_pool': pool_status,
                'avg_operation_times': avg_operation_times,
                'last_health_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取系统健康状态失败: {e}")
            return {}
    
    # ==================== 性能监控和分析 ====================
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            # 操作统计
            total_operations = sum(self._performance_stats['operations_count'].values())
            
            # 平均操作时间
            avg_times = {}
            for op, times in self._performance_stats['operation_times'].items():
                if times:
                    avg_times[op] = {
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'count': len(times)
                    }
            
            # 缓存性能
            cache_stats = self.get_cache_statistics()
            
            # 数据库性能
            db_stats = self._get_database_performance()
            
            return {
                'summary': {
                    'total_operations': total_operations,
                    'initialization_time': self._performance_stats['initialization_time'],
                    'uptime_hours': self._get_uptime_hours()
                },
                'operation_performance': avg_times,
                'cache_performance': cache_stats,
                'database_performance': db_stats,
                'system_health': self._get_system_health(),
                'report_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"生成性能报告失败: {e}")
            return {}
    
    def _get_database_performance(self) -> Dict[str, Any]:
        """获取数据库性能统计"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 表大小统计
                tables = ['query_history', 'document_history', 'system_stats', 'user_sessions', 'batch_operations']
                table_stats = {}
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    table_stats[table] = {'row_count': row_count}
                
                # 索引使用情况
                cursor.execute("PRAGMA index_list('query_history')")
                index_info = cursor.fetchall()
                
                return {
                    'table_statistics': table_stats,
                    'index_count': len(index_info),
                    'database_size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
                }
                
        except Exception as e:
            self.logger.error(f"获取数据库性能统计失败: {e}")
            return {}
    
    def _get_uptime_hours(self) -> float:
        """获取系统运行时间（小时）"""
        try:
            init_time = datetime.fromisoformat(self._performance_stats['initialization_time'])
            uptime = datetime.now() - init_time
            return uptime.total_seconds() / 3600
        except:
            return 0.0
    
    # ==================== 数据导出和备份 ====================
    
    def export_data(self, export_path: Optional[str] = None, include_cache: bool = False) -> str:
        """导出所有数据到指定路径"""
        start_time = time.time()
        
        try:
            # 确定导出路径
            if export_path is None:
                export_path = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            export_dir = Path(export_path)
            export_dir.mkdir(exist_ok=True)
            
            # 导出数据库
            db_export_path = export_dir / "database.db"
            shutil.copy2(self.db_path, db_export_path)
            
            # 导出向量存储
            vector_export_dir = export_dir / "vector_store"
            if self.vector_store_path.exists():
                shutil.copytree(self.vector_store_path, vector_export_dir, dirs_exist_ok=True)
            
            # 导出缓存（可选）
            if include_cache and self.cache_path.exists():
                cache_export_dir = export_dir / "cache"
                shutil.copytree(self.cache_path, cache_export_dir, dirs_exist_ok=True)
            
            # 导出系统状态
            state_export_path = export_dir / "system_state.json"
            current_state = self.load_system_state()
            current_state['export_metadata'] = {
                'export_time': datetime.now().isoformat(),
                'include_cache': include_cache,
                'performance_stats': self._performance_stats
            }
            
            with open(state_export_path, 'w', encoding='utf-8') as f:
                json.dump(current_state, f, ensure_ascii=False, indent=2)
            
            # 创建导出元数据
            metadata = {
                'export_time': datetime.now().isoformat(),
                'export_version': '1.0',
                'include_cache': include_cache,
                'file_structure': self._get_export_file_structure(export_dir),
                'system_info': self._get_system_health()
            }
            
            metadata_path = export_dir / "export_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            operation_time = time.time() - start_time
            self._record_operation('export_data', operation_time)
            
            self.logger.info(f"数据导出完成: {export_dir}")
            return str(export_dir)
            
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
            raise
    
    def _get_export_file_structure(self, export_dir: Path) -> Dict[str, Any]:
        """获取导出文件结构信息"""
        try:
            structure = {}
            for item in export_dir.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(export_dir)
                    structure[str(relative_path)] = {
                        'size_bytes': item.stat().st_size,
                        'modified_time': item.stat().st_mtime
                    }
            return structure
        except Exception as e:
            self.logger.error(f"获取导出文件结构失败: {e}")
            return {}
    
    def import_data(self, import_path: str, restore_cache: bool = False) -> bool:
        """从导出路径导入数据"""
        start_time = time.time()
        
        try:
            import_dir = Path(import_path)
            if not import_dir.exists():
                raise FileNotFoundError(f"导入路径不存在: {import_path}")
            
            # 验证导入数据
            metadata_path = import_dir / "export_metadata.json"
            if not metadata_path.exists():
                raise ValueError("缺少导出元数据文件")
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.logger.info(f"导入数据版本: {metadata.get('export_version', 'unknown')}")
            
            # 备份现有数据
            backup_path = self._create_backup()
            
            try:
                # 导入数据库
                db_import_path = import_dir / "database.db"
                if db_import_path.exists():
                    shutil.copy2(db_import_path, self.db_path)
                    # 重新初始化连接池
                    self._initialize_connection_pool()
                
                # 导入向量存储
                vector_import_dir = import_dir / "vector_store"
                if vector_import_dir.exists():
                    if self.vector_store_path.exists():
                        shutil.rmtree(self.vector_store_path)
                    shutil.copytree(vector_import_dir, self.vector_store_path)
                
                # 导入缓存（可选）
                if restore_cache:
                    cache_import_dir = import_dir / "cache"
                    if cache_import_dir.exists():
                        if self.cache_path.exists():
                            shutil.rmtree(self.cache_path)
                        shutil.copytree(cache_import_dir, self.cache_path)
                
                # 导入系统状态
                state_import_path = import_dir / "system_state.json"
                if state_import_path.exists():
                    with open(state_import_path, 'r', encoding='utf-8') as f:
                        imported_state = json.load(f)
                    self.save_system_state(imported_state)
                
                operation_time = time.time() - start_time
                self._record_operation('import_data', operation_time)
                
                self.logger.info(f"数据导入完成: {import_path}")
                return True
                
            except Exception as e:
                # 恢复备份
                self.logger.error(f"导入失败，恢复备份: {e}")
                self._restore_from_backup(backup_path)
                raise
                
        except Exception as e:
            self.logger.error(f"数据导入失败: {e}")
            return False
    
    def _create_backup(self) -> str:
        """创建当前数据的备份"""
        backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"backup_{backup_time}"
        return self.export_data(backup_path, True)  # 修正参数传递
    
    def _restore_from_backup(self, backup_path: str):
        """从备份恢复数据"""
        try:
            self.import_data(backup_path, True)  # 修正参数传递
        except Exception as e:
            self.logger.error(f"备份恢复失败: {e}")
    
    # ==================== 清理和维护 ====================
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """清理旧数据"""
        start_time = time.time()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self._transaction() as conn:
                cursor = conn.cursor()
                # 清理旧的查询历史
                cursor.execute(
                    "DELETE FROM query_history WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_queries = cursor.rowcount
                
                # 清理旧的文档历史
                cursor.execute(
                    "DELETE FROM document_history WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_docs = cursor.rowcount
                
                # 清理旧的用户会话
                cursor.execute(
                    "DELETE FROM user_sessions WHERE last_activity < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_sessions = cursor.rowcount
                
                # 清理旧的批处理操作记录
                cursor.execute(
                    "DELETE FROM batch_operations WHERE start_time < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_batches = cursor.rowcount
            
            # 清理过期缓存
            expired_cache_items = self._cleanup_expired_cache()
            
            operation_time = time.time() - start_time
            self._record_operation('cleanup_old_data', operation_time)
            
            cleanup_stats = {
                'queries_deleted': deleted_queries,
                'documents_deleted': deleted_docs,
                'sessions_deleted': deleted_sessions,
                'batches_deleted': deleted_batches,
                'cache_items_deleted': expired_cache_items
            }
            
            self.logger.info(f"数据清理完成: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"数据清理失败: {e}")
            raise
    
    def _cleanup_expired_cache(self) -> int:
        """清理过期缓存项"""
        try:
            deleted_count = 0
            current_time = datetime.now()
            
            # 检查所有缓存文件的过期状态
            cache_files = list(self.cache_path.glob("*.pkl")) + list(self.cache_path.glob("*.gz.pkl"))
            
            for cache_file in cache_files:
                try:
                    # 尝试加载缓存数据检查过期时间
                    if cache_file.name.endswith('.gz.pkl'):
                        with gzip.open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                    else:
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                    
                    # 检查是否过期
                    if 'timestamp' in cache_data and 'expire_hours' in cache_data:
                        cache_time = datetime.fromisoformat(cache_data['timestamp'])
                        expire_hours = cache_data['expire_hours']
                        
                        if (current_time - cache_time).total_seconds() > expire_hours * 3600:
                            cache_file.unlink()
                            deleted_count += 1
                            
                except Exception:
                    # 如果无法读取缓存文件，删除它
                    cache_file.unlink()
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"清理过期缓存失败: {e}")
            return 0
    
    def optimize_database(self):
        """优化数据库性能"""
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 执行VACUUM清理数据库
                cursor.execute("VACUUM")
                
                # 重建索引
                cursor.execute("REINDEX")
                
                # 更新统计信息
                cursor.execute("ANALYZE")
            
            operation_time = time.time() - start_time
            self._record_operation('optimize_database', operation_time)
            
            self.logger.info("数据库优化完成")
            
        except Exception as e:
            self.logger.error(f"数据库优化失败: {e}")
            raise


# 全局持久化管理器实例
persistence_manager = PersistenceManager()
