import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

type Theme = 'light' | 'dark' | 'system';

interface ThemeState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

// 应用主题到 HTML 根节点
function applyTheme(theme: Theme) {
  const root = window.document.documentElement;
  
  // 移除 dark 类
  root.classList.remove('dark');

  let effectiveTheme = theme;
  if (theme === 'system') {
    effectiveTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  // 暗色模式添加 dark 类
  if (effectiveTheme === 'dark') {
    root.classList.add('dark');
  }
  
  console.log('[Theme] Applied:', theme, '-> effective:', effectiveTheme, '| HTML class:', root.className);
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      theme: 'dark',
      setTheme: (theme) => {
        console.log('[Theme] setTheme called with:', theme);
        set({ theme });
        applyTheme(theme);
      },
    }),
    {
      name: 'zoaholic-theme',
      storage: createJSONStorage(() => localStorage),
      onRehydrateStorage: () => (state) => {
        // 在状态从 localStorage 恢复后应用主题
        if (state) {
          console.log('[Theme] Rehydrated from storage:', state.theme);
          applyTheme(state.theme);
        }
      },
    }
  )
);

// 监听 store 变化并应用主题
useThemeStore.subscribe((state) => {
  applyTheme(state.theme);
});

// 初始化时立即应用默认主题（防止闪烁），rehydrate 后会用正确的值覆盖
applyTheme(useThemeStore.getState().theme);

// 监听系统主题变化
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  const store = useThemeStore.getState();
  if (store.theme === 'system') {
    applyTheme('system');
  }
});
