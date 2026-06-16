#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
  void*m=malloc(n);if(fread(m,1,n,f)!=(size_t)n)return 2;fclose(f);
  rknn_context ctx;if(rknn_init(&ctx,m,n,0,NULL))return 1;
  rknn_input_output_num io;rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));
  printf("n_input=%u n_output=%u\n",io.n_input,io.n_output);
  for(unsigned i=0;i<io.n_output;i++){rknn_tensor_attr a;memset(&a,0,sizeof(a));a.index=i;
    rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a));
    printf("out%u: dims=[%u,%u,%u,%u] n_elems=%u type=%d fmt=%d zp=%d scale=%f\n",
      i,a.dims[0],a.dims[1],a.dims[2],a.dims[3],a.n_elems,a.type,a.fmt,a.zp,a.scale);}
  return 0;
}
