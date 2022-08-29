import logging
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, parallel_bulk
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#logger.info(" Establishing Elasticsearch Database ... ")

class ElasticDB(object):

    def __init__(self, elastic_port=None, elastic_index=None):
        """
        Args:
            :param elastic_port: Port for Elasticsearch instance
            :param elastic_index: Index name
            :param elastic_fields: Search fields
        """
        # Connect
        logger.info('Connecting to %s ' % elastic_port)    
        self.elastic_port = elastic_port
        self.es = Elasticsearch(elastic_port, retry_on_timeout=True)
        self.elastic_index = elastic_index

        logger.info('Connected to %s ' % self.es)

    def add_doc(self, doc):
        """ Add doc to DB """
        response = self.es.index(
            index = self.elastic_index,
            document = doc,
        )

        logger.info('Added document id %s' % response["_id"])

    def bulk_add(self, index_name, source, iterator, chunk_size):
        
        errors_before_interrupt = 5
        successes = 0
        
        for ok, result in parallel_bulk(self.es, iterator(idx=index_name, source=source), chunk_size=chunk_size, request_timeout=60*3):
            if ok is not True:
                logger.error('Failed to import data')
                logger.error(str(result))
                errors_count += 1

            if errors_count == errors_before_interrupt:
                logging.fatal('Too many import errors, exiting with error code')
                exit(1)
            
            successes += ok


    def search(self, query_, k=5):
        results = self.es.search(
            index = self.elastic_index,
            query = {
                "match": {
                    "document.text": query_,
                    },
            },
            size=k)

        hits = results["hits"]["hits"]
        #doc_ids = [row['_source']["document"]["id"] for row in hits]

        return hits

        # results = self.es.search(
        #     index=self.elastic_index,
        #     body= {
        #         "size": k,
        #         "query": {
        #             "match": {
        #                 "text": query,
        #     }}})

        # hits = results["hits"]["hits"]
        # doc_ids = [row['_source']["id"] for row in hits]

        # return (hits, doc_ids)

    # Elasticsearch Controls

    def __enter__(self):
        return self

    def __close__(self):
        self.es = None

    # def get_doc_index(self, doc_id, field_doc_name):
    #     """Convert doc_id --> doc_index"""
    #     field_index = field_doc_name
    #     if isinstance(field_index, list):
    #         field_index = '.'.join(field_index)
    #     result = self.es.search(index=self.elastic_index, body={'query':{'match':
    #         {field_index: doc_id}}})
    #     return result['hits']['hits'][0]['_id']

    # def get_doc_text(self, doc_id):
    #     """ Fetch doc text for given doc_id """
    #     idx = self.get_doc_index(doc_id)



