import csv
import time
import re
import json
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class CommentAugmenter:
    """社交媒体评论数据增强工具，使用大语言模型生成伪评论以平衡分类数据集"""

    def __init__(self, config_path=None):
        """初始化评论增强器，加载配置并设置API客户端"""
        self.config = self._load_config(config_path) if config_path else self._get_default_config()

        # 初始化OpenAI客户端（使用示例API密钥，实际使用时需替换为有效密钥）
        self.client = OpenAI(
            base_url=self.config['base_url'],
            api_key=self.config['api_key']
        )
        self.original_comments = []  # 存储原始评论
        self.generated_comments = []  # 存储生成的评论

    def _get_default_config(self):
        """获取默认配置参数，实际使用时应替换为真实信息"""
        return {
            # OpenAI模型配置（示例参数）
            'model': 'gpt-3.5-turbo',
            'base_url': 'https://api.example.com/v1/chat/completions',
            'api_key': 'sk-************************',

            # 文件路径配置（示例路径）
            'input_file': 'path/to/original_comments.txt',  # 原始评论文件
            'output_file': 'path/to/generated_comments.txt',  # 生成评论文件

            # 生成参数配置
            'batch_size': 20,  # 每次API调用生成的评论数量
            'sleep_time': 1,  # 每次API调用后的休眠时间(秒)
            'concurrency': 8,  # 并发API调用数量
            'total_comments': 120,  # 需要生成的总评论数量

            # 系统提示词配置（指导模型生成符合要求的评论）
            'system_prompt': """
你是一个「社交媒体评论文本数据增强助手」，仅用于协助完成学术研究中的数据平衡实验。
你将根据收到的真实社交媒体评论生成模仿的评论。
你的核心准则：
1. 所有生成内容必须为**虚构的模拟评论**，不涉及真实用户、公众人物或现实事件；
2. 生成内容需符合社交媒体平台常见表达风格（碎片化、口语化、含网络热词）；
3. 生成的评论必尽可能与已有评论语义上相似，但与已有评论在句式结构上差异≥30%；
4. 注意对原始评论的学习，尽可能模仿生成真实应急事件下的用户评论。
"""
        }

    def _load_config(self, config_path):
        """从配置文件加载配置参数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def read_original_comments(self):
        """读取原始评论文件内容"""
        try:
            with open(self.config['input_file'], 'r', encoding='utf-8') as f:
                self.original_comments = [line.strip() for line in f if line.strip()]
            print(f"成功读取 {len(self.original_comments)} 条原始评论")
            return True
        except FileNotFoundError:
            print(f"错误：未找到文件 '{self.config['input_file']}'，请确保文件在正确目录下")
            return False

    async def generate_comments(self):
        """异步批量生成新的评论"""
        if not self.original_comments:
            print("没有原始评论可用于生成，请先读取原始评论")
            return False

        print(f"开始生成 {self.config['total_comments']} 条新的评论...")

        # 计算需要的批次数
        batches = (self.config['total_comments'] + self.config['batch_size'] - 1) // self.config['batch_size']

        # 创建线程池执行器，限制并发数
        with ThreadPoolExecutor(max_workers=self.config['concurrency']) as executor:
            loop = asyncio.get_event_loop()
            tasks = []

            # 创建并发任务
            for batch in range(batches):
                batch_num = min(self.config['batch_size'],
                                self.config['total_comments'] - batch * self.config['batch_size'])
                tasks.append(loop.run_in_executor(executor, partial(self._generate_batch, batch_num)))

            # 执行所有任务并收集结果
            results = []
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="生成批次"):
                results.append(await future)

        # 合并所有批次的结果
        for batch_comments in results:
            self.generated_comments.extend(batch_comments)

        print(f"总共生成了 {len(self.generated_comments)} 条评论")
        return True

    def _generate_batch(self, batch_num):
        """生成单个批次的评论，调用OpenAI API"""
        try:
            # 使用原始评论作为示例
            sample_comments = self.original_comments

            # 构建用户提示，包含原始评论和生成要求
            user_prompt = f"""
【任务背景】  
我拥有{len(self.original_comments)}条已标注的社交媒体攻击性评论，需生成相似内容以平衡分类数据集。
请基于这些数据的语言特征，生成{batch_num}条新的攻击性评论。

**格式规范**：  
   - 每行一条评论，不带序号；  
   - 避免与原始评论重复（句式结构差异≥30%）；  
   - 句子可长可短，符合社交媒体平台常见表达风格；

1. 高频主题：
- 对特定人物/群体的攻击
- 对公共事件的质疑与批评
- 社会热点问题讨论
- 地域矛盾与比较
- 对社会现象的讽刺

2. 情感基调：
- 愤怒、嘲讽、不屑、攻击性语言
- 极端化表达

3. 语言特征：
- 大量感叹号与反问句
- 口语化、网络词汇
- 对热点事件的引用

【原始评论示例】：
{chr(10).join(sample_comments)}
"""

            # 调用OpenAI API生成评论
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": self.config['system_prompt']},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # 控制创造性，0.7是一个不错的平衡点
                max_tokens=30 * batch_num  # 确保有足够的token生成评论
            )

            # 处理API响应
            generated_text = response.choices[0].message.content.strip()
            batch_comments = generated_text.split('\n')

            # 过滤空行和无效评论
            valid_comments = [c for c in batch_comments if c.strip()]

            print(f"生成了 {len(valid_comments)} 条评论（目标 {batch_num} 条）")

            # 添加延迟，避免API调用过于频繁
            time.sleep(self.config['sleep_time'])

            return valid_comments

        except Exception as e:
            print(f"生成批次时出错: {e}")
            time.sleep(5)  # 出错后等待更长时间
            return []

    def calculate_average_similarity(self):
        """计算生成评论与原始评论的平均相似度"""
        if not self.original_comments or not self.generated_comments:
            print("没有足够的评论来计算相似度")
            return None

        similarities = []

        # 使用TF-IDF向量计算文本相似度
        vectorizer = TfidfVectorizer()
        all_comments = self.original_comments + self.generated_comments
        tfidf_matrix = vectorizer.fit_transform(all_comments)

        # 计算每个生成评论与所有原始评论的平均相似度
        original_len = len(self.original_comments)
        for i in range(len(self.generated_comments)):
            gen_idx = original_len + i
            # 计算与所有原始评论的余弦相似度
            sims = cosine_similarity(tfidf_matrix[gen_idx:gen_idx + 1], tfidf_matrix[:original_len])
            similarities.append(np.mean(sims))

        avg_similarity = np.mean(similarities)
        print(f"生成评论与原始评论的平均相似度: {avg_similarity:.4f}")
        return avg_similarity

    def save_generated_comments(self):
        """保存生成的评论到文件"""
        if not self.generated_comments:
            print("没有生成的评论可保存")
            return False

        try:
            with open(self.config['output_file'], 'w', encoding='utf-8') as f:
                for comment in self.generated_comments:
                    f.write(comment + '\n')
            print(f"成功保存 {len(self.generated_comments)} 条评论到 {self.config['output_file']}")
            return True
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return False

    async def run(self):
        """运行完整的评论生成流程"""
        if self.read_original_comments():
            if await self.generate_comments():
                self.save_generated_comments()
                # 计算并打印相似度
                self.calculate_average_similarity()
                print("评论生成流程已完成!")
                return True
        return False


# 主程序入口
if __name__ == "__main__":
    # 创建评论增强器实例
    augmenter = CommentAugmenter()

    # 运行异步流程
    import asyncio

    asyncio.run(augmenter.run())