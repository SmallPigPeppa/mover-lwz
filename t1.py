'''
class Solution {
    public int compress(char[] cs) {
        int n = cs.length;
        int i = 0, j = 0;
        while (i < n) {
            int idx = i;
            while (idx < n && cs[idx] == cs[i]) idx++;
            int cnt = idx - i;
            cs[j++] = cs[i];
            if (cnt > 1) {
                int start = j, end = start;
                while (cnt != 0) {
                    cs[end++] = (char)((cnt % 10) + '0');
                    cnt /= 10;
                }
                reverse(cs, start, end - 1);
                j = end;
            }
            i = idx;
        }
        return j;
    }
    void reverse(char[] cs, int start, int end) {
        while (start < end) {
            char t = cs[start];
            cs[start] = cs[end];
            cs[end] = t;
            start++; end--;
        }
    }
}
'''


def reverse(cs, start, end):
    while start < end:
        t = cs[start]
        cs[start] = cs[end]
        cs[end] = t
        start += 1
        end -= 1
    return cs


def compress(cs):
    n = len(cs)
    i = 0
    j = 0
    while i < n:
        idx = i;
        while (idx < n and cs[idx] == cs[i]):
            idx += 1
        cnt = idx - i
        j += 1
        cs[j] = cs[i]
        if (cnt > 1):

            start = j
            end = start
            while cnt != 0:
                end += 1
            cs[end] = str((cnt % 10) + '0')
            cnt /= 10;

        cs = reverse(cs, start, end - 1);
        j = end
        i = idx
        return j
# int main() {
#     //   freopen("a.in", "r", stdin);
#     string s;
#     cin >> s;
#     for (;;)
#     {
#         int _st = -1, _l = -1, _cnt = -1;
#         for (int st = 0; st < s.length(); st++)
#             if (s[st] != '(' && s[st] != ')' && !isdigit(s[st]))
#                 for (int l = 1; st + l + l <= s.length(); l++)
#                     if (s.substr(st, l) == s.substr(st + l, l))
#                     {
#                         int cnt = 2, j = st + l + l;
#                         while (j + l <= s.length() && s.substr(j, l) == s.substr(st, l))
#                             j += l, cnt++;
#                         if (l * cnt > _l * _cnt || l * cnt == _l * _cnt && cnt > _cnt)
#                             _cnt = cnt, _l = l, _st = st;
#                     }
#         if (_st == -1)
#             break;
#         string tmp = to_string(_cnt);
#         s = s.substr(0, _st) + tmp + "(" + s.substr(_st, _l) + ")" + s.substr(_st + _cnt * _l);
#         //        cout << s << endl;
#     }
#
#     return cout << s << endl, 0;
# }
if __name__=="__main__":
    s=str(input())
    while True:
        _st = -1
        _l = -1
        _cnt = -1
        for st in range(len(s)):
            if (s[st] != '(' and s[st] != ')' and s[st].isdigit()):
                for l in range(1,int((len(s)-st)/2)):
                    if (s[st, l] == s[st + l, l]):
                        cnt = 2
                        j = st + l + l
                        while (j + l <= len(s) and  s[j, l] == s[st, l]):
                            j += l
                            cnt +=1
                        if (l * cnt > _l * _cnt or l * cnt == _l * _cnt and cnt > _cnt):
                            _cnt = cnt
                            _l = l
                            _st = st
        if (_st == -1):
            break
    tmp = str(_cnt)
    ans = s[0, int(_st)] + tmp + "(" + s[int(_st), int(_l)] + ")" + s[int(_st) + int(_cnt) * int(_l)]
    print(ans)

