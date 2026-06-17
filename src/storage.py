"""
存储模块
负责事件记录、照片存储和自动清理
"""
import os
import time
import sqlite3
import cv2
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.config import get_config, BASE_DIR


class Storage:
    def __init__(self):
        config = get_config()
        storage_cfg = config.get("storage", {})

        self.sqlite_path = os.path.join(BASE_DIR, storage_cfg.get("sqlite_path", "data/events.db"))
        self.photo_dir = os.path.join(BASE_DIR, storage_cfg.get("photo_dir", "data/photos"))
        self.photo_quality = storage_cfg.get("photo_quality", 90)
        self.max_photo_size_mb = storage_cfg.get("max_photo_size_mb", 1024)
        self.auto_cleanup_enabled = storage_cfg.get("auto_cleanup_enabled", True)
        self.retention_days = config.get("supervision", {}).get("event_retention_days", 30)

        # 确保目录存在
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        os.makedirs(self.photo_dir, exist_ok=True)

        # 初始化数据库
        self._init_db()

        # 上次清理时间
        self.last_cleanup_time = 0
        self.cleanup_interval = 3600  # 每小时清理一次

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        # 创建事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                photo_path TEXT,
                timestamp REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建维测事件表（调试/打点专用，不通知，不影响统计，不影响 events 表 ID 序列）
        # 结构与 events 完全一致，方便 SQL join 分析
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events_debug (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                photo_path TEXT,
                timestamp REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建统计表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_events INTEGER DEFAULT 0,
                notice_count INTEGER DEFAULT 0,
                warning_count INTEGER DEFAULT 0,
                danger_count INTEGER DEFAULT 0,
                cry_count INTEGER DEFAULT 0,
                exposure_count INTEGER DEFAULT 0,
                occlusion_count INTEGER DEFAULT 0,
                region_exit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    @staticmethod
    def _jsonable(obj):
        """递归把含 numpy / tuple 的对象转成 json 可序列化的形式。"""
        try:
            import numpy as _np
        except Exception:
            _np = None
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {str(k): Storage._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Storage._jsonable(v) for v in obj]
        if _np is not None and isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if _np is not None and isinstance(obj, _np.integer):
            return int(obj)
        if _np is not None and isinstance(obj, _np.floating):
            f = float(obj)
            return f if (f == f and f not in (float("inf"), float("-inf"))) else None
        if isinstance(obj, float):
            return obj if (obj == obj and obj not in (float("inf"), float("-inf"))) else None
        if isinstance(obj, (int, str, bool)):
            return obj
        try:
            return str(obj)
        except Exception:
            return None

    def _serialize_details(self, details) -> str:
        """把 details 写入 DB 前序列化。优先 JSON，失败回退 str()（保持向后兼容）。"""
        if details is None:
            return None
        try:
            import json
            return json.dumps(self._jsonable(details), ensure_ascii=False)
        except Exception:
            try:
                return str(details)
            except Exception:
                return None

    def save_photo(self, frame, timestamp: float = None) -> Optional[str]:
        """保存抓拍照片
        返回保存的文件路径
        """
        if timestamp is None:
            timestamp = time.time()

        # 按日期分目录存储
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        date_dir = os.path.join(self.photo_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)

        # 文件名格式: 年月日-时分秒-毫秒.jpg
        time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d-%H%M%S-%f")[:-3]
        filename = f"{time_str}.jpg"
        filepath = os.path.join(date_dir, filename)

        try:
            cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.photo_quality])
            return filepath
        except Exception as e:
            print(f"保存照片失败: {e}")
            return None

    def save_event(self, event_type: str, level: str, message: str,
                   details: Optional[Dict] = None, photo_path: Optional[str] = None,
                   update_stats: bool = True, to_debug_table: bool = False) -> int:
        """保存事件到数据库
        返回事件ID

        update_stats: 是否更新 statistics 表与触发清理。
                      调试/维测打点设为 False，避免污染日统计。
        to_debug_table: 写入 events_debug 维测表（不占用 events 表ID序列）。
                        用于 audio_heartbeat 等频繁打点的维测数据。
        """
        timestamp = time.time()
        # 序列化 details：尽量用 JSON（便于事后查询/解析），失败时回退 str()
        details_str = self._serialize_details(details) if details else None

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        table_name = "events_debug" if to_debug_table else "events"

        try:
            cursor.execute(f'''
                INSERT INTO {table_name} (event_type, level, message, details, photo_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (event_type, level, message, details_str, photo_path, timestamp))

            event_id = cursor.lastrowid

            if update_stats:
                # 更新统计
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                cursor.execute('SELECT id FROM statistics WHERE date = ?', (date_str,))
                if cursor.fetchone():
                    # 更新现有统计
                    update_fields = {
                        "total_events": "total_events + 1",
                        f"{level}_count": f"{level}_count + 1"
                    }

                    # 特定事件计数
                    if event_type == "cry_detected":
                        update_fields["cry_count"] = "cry_count + 1"
                    elif event_type == "limb_exposure":
                        update_fields["exposure_count"] = "exposure_count + 1"
                    elif event_type == "occlusion_detected":
                        update_fields["occlusion_count"] = "occlusion_count + 1"
                    elif event_type == "region_exit":
                        update_fields["region_exit_count"] = "region_exit_count + 1"

                    set_clause = ", ".join([f"{k} = {v}" for k, v in update_fields.items()])
                    cursor.execute(f'''
                        UPDATE statistics SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE date = ?
                    ''', (date_str,))
                else:
                    # 新建统计记录
                    counts = {
                        "notice_count": 1 if level == "notice" else 0,
                        "warning_count": 1 if level == "warning" else 0,
                        "danger_count": 1 if level == "danger" else 0,
                        "cry_count": 1 if event_type == "cry_detected" else 0,
                        "exposure_count": 1 if event_type == "limb_exposure" else 0,
                        "occlusion_count": 1 if event_type == "occlusion_detected" else 0,
                        "region_exit_count": 1 if event_type == "region_exit" else 0,
                    }

                    cursor.execute('''
                        INSERT INTO statistics (
                            date, total_events, notice_count, warning_count, danger_count,
                            cry_count, exposure_count, occlusion_count, region_exit_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        date_str, 1,
                        counts["notice_count"], counts["warning_count"], counts["danger_count"],
                        counts["cry_count"], counts["exposure_count"],
                        counts["occlusion_count"], counts["region_exit_count"]
                    ))

            conn.commit()

            # 定期清理旧数据（仅在更新统计的路径触发，避免高频维测打点反复清理）
            if update_stats and self.auto_cleanup_enabled and time.time() - self.last_cleanup_time > self.cleanup_interval:
                self._cleanup_old_data()
                self.last_cleanup_time = time.time()

            return event_id

        except Exception as e:
            print(f"保存事件失败: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def _cleanup_old_data(self):
        """清理过期数据"""
        print("[Storage] 开始清理过期数据...")

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cutoff_timestamp = cutoff_date.timestamp()
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        try:
            # 删除过期事件
            cursor.execute('DELETE FROM events WHERE timestamp < ?', (cutoff_timestamp,))
            deleted_events = cursor.rowcount

            # 删除过期统计
            cursor.execute('DELETE FROM statistics WHERE date < ?', (cutoff_date_str,))
            deleted_stats = cursor.rowcount

            conn.commit()

            # 删除过期照片目录
            deleted_photos = 0
            freed_space_mb = 0

            for date_dir in os.listdir(self.photo_dir):
                try:
                    dir_date = datetime.strptime(date_dir, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        dir_path = os.path.join(self.photo_dir, date_dir)
                        # 计算目录大小
                        dir_size = sum(os.path.getsize(os.path.join(dir_path, f))
                                       for f in os.listdir(dir_path)
                                       if os.path.isfile(os.path.join(dir_path, f)))
                        freed_space_mb += dir_size / (1024 * 1024)
                        deleted_photos += len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])

                        shutil.rmtree(dir_path)
                except ValueError:
                    # 不是日期格式的目录跳过
                    continue

            print(f"[Storage] 清理完成: 删除 {deleted_events} 条事件, {deleted_stats} 条统计, "
                  f"{deleted_photos} 张照片, 释放空间 {freed_space_mb:.1f}MB")

        except Exception as e:
            print(f"清理旧数据失败: {e}")
            conn.rollback()
        finally:
            conn.close()

    def save_debug_event(self, event_type: str, level: str, message: str,
                         details: Optional[Dict] = None, photo_path: Optional[str] = None) -> int:
        """保存维测打点事件到 events_debug 表。
        - 不影响 events 表的 ID 序列
        - 不更新 statistics
        - 不触发清理
        - 不发通知
        专门用于 region_debug / 算法中间值记录，便于事后定位误报。
        """
        timestamp = time.time()
        details_str = self._serialize_details(details) if details else None

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO events_debug (event_type, level, message, details, photo_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (event_type, level, message, details_str, photo_path, timestamp))
            event_id = cursor.lastrowid
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return event_id

    def get_events(self, start_time: float = None, end_time: float = None,
                   level: str = None, event_type: str = None, limit: int = 100) -> List[Dict]:
        """查询事件记录"""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = 'SELECT * FROM events WHERE 1=1'
        params = []

        if start_time is not None:
            query += ' AND timestamp >= ?'
            params.append(start_time)

        if end_time is not None:
            query += ' AND timestamp <= ?'
            params.append(end_time)

        if level is not None:
            query += ' AND level = ?'
            params.append(level)

        if event_type is not None:
            query += ' AND event_type = ?'
            params.append(event_type)

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            event = dict(row)
            event['datetime'] = datetime.fromtimestamp(event['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            events.append(event)

        conn.close()
        return events

    def get_statistics(self, date: str = None) -> Optional[Dict]:
        """获取统计数据"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM statistics WHERE date = ?', (date,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return dict(row)
        return None
